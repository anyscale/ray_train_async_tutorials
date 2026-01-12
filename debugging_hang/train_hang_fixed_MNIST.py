# train_hang_fixed.py
import os
import time
import shutil
import tempfile
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torchvision.datasets import MNIST
from torchvision.models import resnet18

import ray
import ray.data as rd
import ray.train as train
import ray.train.torch
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer


# ----------------------------
# Shared storage locations
# ----------------------------
CLUSTER_STORAGE_ROOT = Path("/mnt/cluster_storage")
DATA_ROOT = CLUSTER_STORAGE_ROOT / "data"
LAB_ROOT = CLUSTER_STORAGE_ROOT / "ray_train_hang_lab"
PARQUET_ROOT = LAB_ROOT / "mnist_parquet"
TRAIN_PARQUET = PARQUET_ROOT / "train"
VAL_PARQUET = PARQUET_ROOT / "val"


# ----------------------------
# Model (MNIST + ResNet18)
# ----------------------------
def build_model() -> nn.Module:
    model = resnet18(num_classes=10)
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=(7, 7),
        stride=(2, 2),
        padding=(3, 3),
        bias=False,
    )
    return model


# ----------------------------
# Parquet helpers
# ----------------------------
def _has_parquet_files(dirpath: Path) -> bool:
    return dirpath.exists() and any(dirpath.rglob("*.parquet"))


def _parquet_looks_valid(train_path: Path, val_path: Path) -> bool:
    """Fast-ish validation: schema + rowcount sanity."""
    try:
        if not (_has_parquet_files(train_path) and _has_parquet_files(val_path)):
            return False

        # Read minimal metadata by loading datasets (count will scan, but MNIST is small).
        tds = rd.read_parquet(str(train_path))
        vds = rd.read_parquet(str(val_path))

        cols = set(tds.schema().names)
        if not {"data", "label"}.issubset(cols):
            return False

        # MNIST should be 60k / 10k
        if tds.count() < 50000:
            return False
        if vds.count() < 8000:
            return False

        return True
    except Exception:
        return False


def write_mnist_parquet(force: bool = False) -> None:
    PARQUET_ROOT.mkdir(parents=True, exist_ok=True)

    if not force and _parquet_looks_valid(TRAIN_PARQUET, VAL_PARQUET):
        print(f"[data] Using existing Parquet at {PARQUET_ROOT}")
        return

    # If something exists but is wrong, blow it away (common when iterating quickly).
    if PARQUET_ROOT.exists():
        shutil.rmtree(PARQUET_ROOT, ignore_errors=True)
    TRAIN_PARQUET.mkdir(parents=True, exist_ok=True)
    VAL_PARQUET.mkdir(parents=True, exist_ok=True)

    print(f"[data] Writing MNIST Parquet to {PARQUET_ROOT} ...")

    train_torch = MNIST(root=str(DATA_ROOT), train=True, download=True)
    val_torch = MNIST(root=str(DATA_ROOT), train=False, download=True)

    # IMPORTANT:
    # Use column name "data" (matches torchvision MNIST .data) and "label".
    # Each row's "data" is a (28, 28) uint8 numpy array.
    train_df = pd.DataFrame(
        {
            "data": list(train_torch.data.numpy()),  # list of (28, 28) uint8 arrays
            "label": train_torch.targets.numpy().astype(np.int64),
        }
    )
    val_df = pd.DataFrame(
        {
            "data": list(val_torch.data.numpy()),
            "label": val_torch.targets.numpy().astype(np.int64),
        }
    )

    rd.from_pandas(train_df).write_parquet(str(TRAIN_PARQUET))
    rd.from_pandas(val_df).write_parquet(str(VAL_PARQUET))

    print("[data] Parquet write complete.")


# ----------------------------
# Distributed metric helpers
# ----------------------------
def _allreduce_sum(t: torch.Tensor) -> torch.Tensor:
    if train.get_context().get_world_size() == 1:
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def _allreduce_mean(t: torch.Tensor) -> torch.Tensor:
    if train.get_context().get_world_size() == 1:
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= train.get_context().get_world_size()
    return t


# ----------------------------
# Train loop (best-practice pattern)
# ----------------------------
def train_loop_per_worker(config: dict) -> None:
    ctx = train.get_context()
    rank = ctx.get_world_rank()
    world_size = ctx.get_world_size()

    device = ray.train.torch.get_device()

    model = build_model()
    model = ray.train.torch.prepare_model(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config["lr"]))

    global_batch_size = int(config["batch_size"])
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"Global batch size ({global_batch_size}) must be divisible by world size ({world_size})."
        )
    per_worker_batch_size = global_batch_size // world_size

    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    # CRITICAL: dtypes keys MUST match dataset column names (here: "data", "label")
    train_iter = train_shard.iter_torch_batches(
        batch_size=per_worker_batch_size,
        dtypes={"data": torch.uint8, "label": torch.long},
        device=device,
    )
    val_iter = val_shard.iter_torch_batches(
        batch_size=per_worker_batch_size,
        dtypes={"data": torch.uint8, "label": torch.long},
        device=device,
    )

    sleep_per_step_s = float(config.get("sleep_per_step_s", 0.0))

    # ---- 1 epoch ----
    model.train()
    steps = 0
    running_loss = 0.0

    for batch in train_iter:
        x = batch["data"].float().div_(255.0)
        x = (x - 0.5) / 0.5
        x = x.unsqueeze(1)  # (B, 1, 28, 28)

        y = batch["label"]

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        steps += 1
        running_loss += float(loss.detach().item())

        if rank == 0 and steps % 50 == 0:
            print({"epoch": 0, "step": steps, "loss": float(loss.detach().item())})

        if sleep_per_step_s > 0:
            time.sleep(sleep_per_step_s)

    # ---- Validation (global accuracy) ----
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch in val_iter:
            x = batch["data"].float().div_(255.0)
            x = (x - 0.5) / 0.5
            x = x.unsqueeze(1)

            y = batch["label"]
            pred = model(x).argmax(dim=1)

            correct += (pred == y).sum().float()
            total += torch.tensor(float(y.numel()), device=device)

    correct = _allreduce_sum(correct)
    total = _allreduce_sum(total)
    val_acc = (correct / total).item() if total.item() > 0 else 0.0

    avg_loss_local = torch.tensor(running_loss / max(steps, 1), device=device)
    avg_loss = _allreduce_mean(avg_loss_local).item()

    # ---- Report + checkpoint (docs pattern) ----
    checkpoint = None
    with tempfile.TemporaryDirectory() as tmpdir:
        if rank == 0:
            state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save(state_dict, os.path.join(tmpdir, "model.pt"))
            checkpoint = train.Checkpoint.from_directory(tmpdir)

        metrics = {
            "epoch": 0,
            "train_steps": steps,
            "loss": float(avg_loss),
            "val_acc": float(val_acc),
        }

        # IMPORTANT: every rank calls report, with identical metrics.
        train.report(metrics=metrics, checkpoint=checkpoint)


def main() -> None:
    ray.init(address="auto")

    # If your parquet dir got into a weird state, this will cleanly rewrite it.
    write_mnist_parquet(force=False)

    train_ds = rd.read_parquet(str(TRAIN_PARQUET))
    val_ds = rd.read_parquet(str(VAL_PARQUET))

    run_name = f"train_hang_fixed_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_config = RunConfig(
        name=run_name,
        storage_path=str(LAB_ROOT),
    )

    scaling_config = ScalingConfig(num_workers=2, use_gpu=True, accelerator_type="A10G")

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "lr": 1e-3,
            "batch_size": 256,
            "sleep_per_step_s": 0.0,
        },
        scaling_config=scaling_config,
        run_config=run_config,
        datasets={"train": train_ds, "val": val_ds},
    )

    result = trainer.fit()

    # Best practice: use the Result object.
    print("Final metrics:", result.metrics)
    print("Result path:", result.path)

    if result.checkpoint:
        with result.checkpoint.as_directory() as ckpt_dir:
            print("Checkpoint contains:", os.listdir(ckpt_dir))


if __name__ == "__main__":
    main()
