# Runpod Training Guide

This repository now supports two training modes:

- Local-safe mode via `configs/mullivc_config.yaml`
- Cloud-oriented mode via `configs/mullivc_runpod.yaml`

## Recommended Runpod target

Use a GPU with at least 24 GB VRAM for the first meaningful run.

Good options:

- RTX 4090
- A10 24GB
- A5000 24GB
- L4 24GB

Avoid very small GPUs. The local MX250-class path is only suitable for smoke tests.

## Before you start

1. Clone the repository and create the Python 3.13 environment.
2. Make sure you have enough disk space for Hugging Face dataset caching.
3. Set `HF_TOKEN` if you want faster and less rate-limited model and dataset downloads.
4. Optionally set up W&B if you want experiment tracking.
5. Expect the first waveform synthesis call to download the pretrained HiFi-GAN vocoder into `pretrained_models/`.

## Sanity run

Run a one-epoch sanity check before any longer job:

```bash
.venv/bin/python train.py \
  --config configs/mullivc_runpod.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 64 \
  --steps-per-epoch 10 \
  --validation-steps 2
```

This should produce:

- `checkpoints/checkpoint_epoch_0_step_0.pt`
- `checkpoints/best_model.pt`

## First pilot run

Use a bounded subset first to validate loss curves and checkpoint quality:

```bash
.venv/bin/python train.py \
  --config configs/mullivc_runpod.yaml \
  --epochs 5 \
  --max-train-samples 10000 \
  --max-val-samples 512 \
  --steps-per-epoch 500 \
  --validation-steps 50
```

## Longer run

After the pilot looks healthy, remove or increase the sample caps:

```bash
.venv/bin/python train.py \
  --config configs/mullivc_runpod.yaml \
  --epochs 20 \
  --validation-steps 100
```

If memory becomes tight, reduce `--batch-size` to `1`.

## Using the checkpoint

After training, convert audio with:

```bash
.venv/bin/python inference.py \
  --config configs/mullivc_runpod.yaml \
  --checkpoint checkpoints/best_model.pt \
  --source_audio path/to/source.wav \
  --target_speaker_audio path/to/target.wav \
  --output converted.wav
```

## Current limitations

- Cycle consistency is enabled by default and uses real generated waveforms from the configured HiFi-GAN vocoder.
- Inference and demo conversion use the same pretrained HiFi-GAN vocoder path as training.
- Final conversion quality still depends on the quality of the learned mel predictions and the match between your mel settings and the chosen pretrained vocoder.