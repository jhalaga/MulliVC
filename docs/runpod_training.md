# Runpod Training Guide

This repository now supports two training modes:

- Local-safe mode via `configs/mullivc_config.yaml`
- Cloud-oriented mode via `configs/mullivc_runpod.yaml`

It also supports a serverless Flash-based smoke run via `scripts/flash_smoke_train.py`.

## Runpod Flash

Runpod Flash is suitable for this repository's bounded smoke training run.

Why it fits:

- Flash queue-based endpoints are explicitly designed for batch jobs and long-running computations.
- The smoke run in this repository is bounded to a small sample set and a small number of steps.
- The Flash endpoint added in `scripts/flash_smoke_train.py` returns the remote subprocess result, log tails, and checkpoint file metadata.

Authentication options:

- Browser flow: `.venv/bin/flash login`
- API key flow: `export RUNPOD_API_KEY=your_api_key_here`

Local prerequisite:

```bash
uv pip install runpod-flash
```

Optional local-only evaluation dependency:

```bash
.venv/bin/python -m pip install -r requirements-eval.txt
```

Run the Flash smoke training endpoint:

```bash
.venv/bin/python scripts/flash_smoke_train.py
```

Check a submitted Flash job later without keeping the launch terminal open:

```bash
.venv/bin/python scripts/flash_job_control.py status ENDPOINT_ID JOB_ID
```

Cancel a submitted Flash job:

```bash
.venv/bin/python scripts/flash_job_control.py cancel ENDPOINT_ID JOB_ID
```

Useful overrides:

```bash
.venv/bin/python scripts/flash_smoke_train.py \
  --epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 64 \
  --steps-per-epoch 10 \
  --validation-steps 2 \
  --timeout-seconds 3600
```

Notes:

- Remote GPU workers currently run Python 3.12.
- `requirements.txt` is kept Flash-buildable; Whisper-based evaluation stays in `requirements-eval.txt` and is not required for smoke training.
- The Flash smoke endpoint now targets the `AMPERE_24` pool by default because it matches this repository's 24 GB guidance without forcing the scarce `ADA_24` path.
- If Flash reports `no gpu availability` or `workers throttled`, that is a Runpod capacity signal for the selected pool rather than a broken `runpod/flash:py3.12-latest` image tag.
- If you want checkpoint persistence across endpoint recreations, the next step is attaching a Flash network volume.
- The current Flash endpoint disables W&B for the smoke run and forwards `HF_TOKEN` if it is set locally.

## Recommended Runpod target

Use a GPU with at least 24 GB VRAM for the first meaningful run.

Good options:

- RTX 4090
- A10 24GB
- A5000 24GB
- L4 24GB

Avoid very small GPUs. The local MX250-class path is only suitable for smoke tests.

## Flash pod bootstrap

If you want the quickest path on Runpod Flash, create a single GPU pod with SSH enabled and then run:

```bash
curl -fsSL https://raw.githubusercontent.com/jhalaga/MulliVC/ca3d4f76990036ddb729734cde491d516097952c/scripts/runpod_flash_smoke_train.sh -o runpod_flash_smoke_train.sh
bash runpod_flash_smoke_train.sh
```

This script will:

- clone the repository
- check out commit `ca3d4f76990036ddb729734cde491d516097952c`
- install Python 3.13 with `uv`
- create `.venv`
- install `requirements.txt`
- launch the one-epoch smoke run with W&B disabled

Optional environment variables:

```bash
export HF_TOKEN=your_token_here
export REPO_URL=https://github.com/jhalaga/MulliVC.git
export CHECKOUT_REF=ca3d4f76990036ddb729734cde491d516097952c
export WORKDIR=$HOME/MulliVC
export PYTHON_VERSION=3.13
```

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