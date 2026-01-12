# train_hang_fixed.py
import os
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
import ray.data as rd
from ray import train
from ray.train import Checkpoint, RunConfig, ScalingConfig
from ray.train.torch import TorchConfig, TorchTrainer


# ----------------------------
# Synthetic dataset (fast, batched)
# ----------------------------
def make_synth_ds(
    num_samples: int,
    num_classes: int,
    image_shape=(3, 224, 224),
    map_batch_size: int = 256,
):
    ds = rd.range(num_samples)

    def gen_batch(batch: dict) -> dict:
        bsz = len(batch["id"])
        x = np.random.randint(0, 256, size=(bsz, *image_shape), dtype=np.uint8).astype(
            np.float32
        ) / 255.0
        y = np.random.randint(0, num_classes, size=(bsz,), dtype=np.int64)
        return {"image": x, "label": y}

    # Reduce object-store pressure (smaller blocks)
    ctx = rd.DataContext.get_current()
    ctx.target_max_block_size = 32 * 1024 * 1024  # 32MiB

    # NOTE: Ray 2.51 deprecates `concurrency`; keep it simple here.
    ds = ds.map_batches(
        gen_batch,
        batch_size=map_batch_size,
        batch_format="numpy",
        zero_copy_batch=False,
    )
    return ds


# ----------------------------
# Model (small ResNet-ish)
# ----------------------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = None
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.skip is not None:
            x = self.skip(x)
        return F.relu(out + x)


class SmallResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
        self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
        self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
        self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ----------------------------
# Train loop
# ----------------------------
def train_loop_per_worker(config: dict):
    # Stable run id shared via config (same on all ranks)
    run_id = config["run_id"]

    world_rank = train.get_context().get_world_rank()
    world_size = train.get_context().get_world_size()

    torch.manual_seed(config["seed"] + world_rank)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SmallResNet(config["num_classes"]).to(device)
    model = train.torch.prepare_model(model)  # DDP wrap

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # IMPORTANT: shard datasets via Train (no manual split)
    train_shard = train.get_dataset_shard("train")
    val_shard = train.get_dataset_shard("val")

    train_iter = train_shard.iter_torch_batches(
        batch_size=config["per_worker_batch_size"],
        dtypes={"image": torch.float32, "label": torch.long},
        device=device,
        drop_last=True,
    )
    val_iter = val_shard.iter_torch_batches(
        batch_size=config["per_worker_batch_size"],
        dtypes={"image": torch.float32, "label": torch.long},
        device=device,
        drop_last=True,
    )

    # ---- training ----
    model.train()
    step = 0
    loss_val = None
    running = 0.0
    report_every = config["report_every"]

    for batch in train_iter:
        x = batch["image"]
        y = batch["label"]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_val = float(loss.detach().cpu().item())
        running += loss_val
        step += 1

        if world_rank == 0 and step % report_every == 0:
            print({"step": step, "loss": running / report_every})
            running = 0.0

        if step >= config["max_train_steps"]:
            break

    # ---- validation ----
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        vstep = 0
        for batch in val_iter:
            x = batch["image"]
            y = batch["label"]
            with torch.autocast("cuda", enabled=torch.cuda.is_available(), dtype=torch.float16):
                logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            vstep += 1
            if vstep >= config["max_val_steps"]:
                break

    val_acc = correct / max(total, 1)

    metrics = {
        "epoch": 0,
        "train_steps": step,
        "loss": float(loss_val if loss_val is not None else 0.0),
        "val_acc": float(val_acc),
        "global_batch_size": int(config["per_worker_batch_size"] * world_size),
        "per_worker_batch_size": int(config["per_worker_batch_size"]),
        "world_size": int(world_size),
    }

    # ---- checkpoint (rank 0 only, stable path) ----
    ckpt = None
    if world_rank == 0:
        ckpt_dir = Path(f"/tmp/ray_train_ckpt_{run_id}") / "checkpoint_rank0"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")
        ckpt = Checkpoint.from_directory(str(ckpt_dir))

    # Report from all ranks; only rank0 includes checkpoint
    train.report(metrics, checkpoint=ckpt)


# ----------------------------
# Main
# ----------------------------
def main():
    ray.init(address="auto")

    num_workers = 2
    use_gpu = True

    num_classes = 10
    per_worker_bs = 256
    max_train_steps = 200
    max_val_steps = 20

    # Ensure enough data per worker shard
    global_bs = per_worker_bs * num_workers
    train_samples = global_bs * max_train_steps * 2
    val_samples = global_bs * max_val_steps * 2

    train_ds = make_synth_ds(train_samples, num_classes=num_classes, map_batch_size=256)
    val_ds = make_synth_ds(val_samples, num_classes=num_classes, map_batch_size=256)

    run_id = uuid.uuid4().hex[:10]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"/mnt/cluster_storage/ray_train_hang_lab/train_hang_fixed_{ts}_{run_id}"

    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "run_id": run_id,
            "seed": 1234,
            "num_classes": num_classes,
            "per_worker_batch_size": per_worker_bs,
            "max_train_steps": max_train_steps,
            "max_val_steps": max_val_steps,
            "lr": 3e-4,
            "report_every": 20,
        },
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu, accelerator_type="A10G"),
        run_config=RunConfig(storage_path=out_dir),
        torch_config=TorchConfig(backend="nccl" if use_gpu else "gloo"),
        datasets={"train": train_ds, "val": val_ds},
    )

    result = trainer.fit()

    df = pd.DataFrame([result.metrics])
    print("\nFinal metrics (DataFrame):")
    print(df)

    print(f"\nResult path: {out_dir}")
    if result.checkpoint:
        ckpt_local = result.checkpoint.to_directory()
        print("Checkpoint materialized to:", ckpt_local)
        print("Files:", os.listdir(ckpt_local))


if __name__ == "__main__":
    main()
