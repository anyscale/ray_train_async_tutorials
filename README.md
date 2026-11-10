# Advanced Ray Train on Anyscale

A hands-on tutorial on removing the bottlenecks that waste GPU time in
distributed training: failure recovery, checkpoint uploads, and validation.

It extends the
[introductory Ray Train tutorial](https://console.anyscale.com/template-preview/ai-libraries-pytorch?file=%252Ffiles%252F2_Intro_Train.ipynb)
and trains a ResNet-18 on MNIST, so that the focus stays on the training
system and not on the model.

## What it covers

| Feature | What you do in the notebook | Why it matters |
|---|---|---|
| **Mid-epoch resumption** | Checkpoint the Ray Data iterator with the model, crash a worker mid-epoch, and resume from the last checkpoint instead of the start of the epoch. | A failure costs the batches since the last checkpoint, not the whole epoch. |
| **Asynchronous checkpointing** | Report checkpoints with `CheckpointUploadMode.ASYNC`. | Uploads to shared storage don't block training. |
| **Asynchronous validation** | Register a `ValidationConfig` and validate each epoch's checkpoint with Ray Data on a separate GPU. | Training GPUs keep training while validation runs. |

Mid-epoch resumption is the centerpiece. The notebook kills a training worker
with `SIGKILL` at batch 120 of an epoch, 20 batches after a checkpoint, and
shows the run resuming at batch 100:

```text
[rank 0] epoch 1 | batch 100 | mid-epoch checkpoint
[rank 1] epoch 1 | batch 120 | WORKER CRASH (simulated with SIGKILL)
[rank 0] epoch 1 | batch 100 | RESUMED from checkpoint. Skipping the 12,800 rows this worker already trained on in this epoch.
[rank 0] epoch 1 | batch 150 | mid-epoch checkpoint
```

It then verifies from the reported metrics that the interrupted epoch covered
the full training set, the same as the uninterrupted epochs.

## Requirements

- An [Anyscale](https://www.anyscale.com/) workspace. Mid-epoch resumption is
  part of Anyscale Runtime and isn't available in open source Ray.
- The `anyscale/ray:2.60.0-py311-cu128` base image (Ray 2.60.0, Python 3.11).
- **3 GPUs** of any type (for example T4, L4, or A10G): 2 for the training
  workers and 1 for asynchronous validation.
- Shared cluster storage at `/mnt/cluster_storage`, which Anyscale workspaces
  mount by default.

## Setup

Use either option.

**Install into a running workspace.** On an Anyscale workspace, this also
registers the packages on every node in the cluster:

```bash
pip install -r requirements.txt
```

**Build a custom image.** The `Dockerfile` starts from the base image above and
installs the same pinned packages.

## Run it

Open `ray_train_new_features_final_draft.ipynb` in the workspace and run the
cells from top to bottom. The whole notebook takes a few minutes; the training
run itself takes about two minutes on T4 GPUs.

The notebook is committed with its outputs, so you can also read it without
running anything.

All artifacts go under `/mnt/cluster_storage/ray_train_async_tutorial`. The
last code cell deletes that directory.

## Repository layout

| Path | Description |
|---|---|
| `ray_train_new_features_final_draft.ipynb` | The tutorial notebook, with outputs. |
| `requirements.txt` | Pinned Python dependencies on top of the base image. |
| `Dockerfile` | Custom image definition: the base image plus `requirements.txt`. |
| `debugging_hang/` | Separate scripts from a training-hang debugging exercise. They aren't part of this tutorial. |
