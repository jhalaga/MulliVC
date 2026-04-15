# MulliVC Training — Monitoring Guide

## Your Active Training Job

| Detail | Value |
|---|---|
| Endpoint ID | `nivp8gwhwplup3` |
| Job ID | `e3e5eecd-3494-483d-8c28-32749efd5eb0-e2` |
| Run ID | `577788cf` |
| GPU | NVIDIA Ada 80 GB (ADA_80_PRO) |
| Config | 20 epochs, 7000 steps/epoch, 200 validation steps, batch size 2 |
| Max runtime | ~23 hours (will stop automatically) |

---

## How to Check Status

Run this from the project folder at any time:

```bash
.venv/bin/python scripts/check_job.py nivp8gwhwplup3 e3e5eecd-3494-483d-8c28-32749efd5eb0-e2
```

If you are connected to the worker web terminal, you can see live training progress directly:

```bash
python /tmp/MulliVC/scripts/monitor_training_progress.py /tmp/MulliVC/flash_runs/577788cf
tail -f /tmp/MulliVC/flash_runs/577788cf/logs/training.log
```

The most useful monitoring files on the worker are:

| File | What it gives you |
|---|---|
| `/tmp/MulliVC/flash_runs/577788cf/logs/progress.json` | Current epoch, current batch, best val loss, latest checkpoint |
| `/tmp/MulliVC/flash_runs/577788cf/logs/metrics.jsonl` | Structured step and epoch metrics |
| `/tmp/MulliVC/flash_runs/577788cf/logs/training.log` | Human-readable training output |

Possible statuses:

| Status | Meaning |
|---|---|
| `IN_QUEUE` | Waiting for a GPU worker. Usually <2 minutes. |
| `IN_PROGRESS` | Training is running. This is the normal state for hours. |
| `COMPLETED` | Done! Check the JSON output for results. |
| `FAILED` | Something crashed. Check the error message. |
| `TIMED_OUT` | Ran over the 24h execution limit. |

---

## What the Output Means (When It Finishes)

When the job completes, you'll see a JSON block. Here's what matters:

```json
{
  "ok": true,              // true = training finished without crashing
  "returncode": 0,         // 0 = success, anything else = error
  "duration_seconds": ..., // how long it took
  "checkpoint_files": [...],  // saved model weights
  "stdout_tail": [...],    // last lines of training output
  "stderr_tail": [...]     // warnings and errors (some are OK)
}
```

### The Key Line in `stdout_tail`

Look for a line like:

```
Device: cuda | batch_size=2 | max_train_samples=None | ...
```

- **`Device: cuda`** = GPU is being used (good!)
- **`Device: cpu`** = fell back to CPU (very slow, something is wrong)

### Training Progress Lines

After each epoch you'll see:

```
Epoch 0:
  Train Loss: 33.9581
  Val Loss: 61.3623
Epoch 1:
  Train Loss: 28.1234
  Val Loss: 45.6789
...
Training completed!
```

---

## Understanding the Numbers

### What Is "Loss"?

Loss is a number that tells you **how wrong the model is**. Think of it as a score in golf — **lower is better**.

- **Train Loss** — how wrong the model is on data it's learning from
- **Val Loss (Validation Loss)** — how wrong the model is on data it has **never seen**

Val Loss is the more important number because it tells you if the model is actually learning to generalize, not just memorizing.

### What Is G_Loss and D_Loss?

MulliVC uses a GAN (Generative Adversarial Network). Think of it as two players:

- **G (Generator)** — tries to create realistic voice conversions
- **D (Discriminator)** — tries to tell real audio from fake audio

They compete against each other, which makes both better over time.

- **G_Loss** — how badly the generator is fooling the discriminator (lower = generator is doing better)
- **D_Loss** — how well the discriminator can tell real from fake

Both losses jumping around is **normal**. They're in a tug-of-war.

### The Sub-Losses

The training tracks several specific loss components:

| Loss | What It Measures |
|---|---|
| `reconstruction` | How close the generated audio is to the original |
| `timbre` | How well the voice identity (tone/color) is preserved |
| `pitch` | How accurate the pitch/intonation is |
| `asr` | How well spoken content is preserved (can you understand the words?) |
| `adversarial` | How realistic the generated audio sounds |

---

## What "Good" Training Looks Like

### Healthy Signs

1. **Both Train Loss and Val Loss go down** over the first several epochs
2. **Val Loss goes down alongside Train Loss** (not diverging)
3. **G_Loss and D_Loss stay in a similar range** — neither one "wins" completely
4. **Checkpoint files are being saved** after each epoch

### Example of Healthy Progress

```
Epoch 0:  Train Loss: 35.0    Val Loss: 60.0
Epoch 1:  Train Loss: 28.0    Val Loss: 48.0    ← both dropping, great
Epoch 2:  Train Loss: 24.0    Val Loss: 42.0    ← still improving
Epoch 5:  Train Loss: 18.0    Val Loss: 35.0    ← good trend
Epoch 10: Train Loss: 12.0    Val Loss: 28.0    ← slower improvement is normal
Epoch 15: Train Loss: 10.0    Val Loss: 27.5    ← starting to flatten
Epoch 19: Train Loss: 9.0     Val Loss: 27.0    ← nearly converged
```

---

## Warning Signs (When Something Might Be Wrong)

### 1. Val Loss Goes UP While Train Loss Goes DOWN = Overfitting

```
Epoch 5:  Train Loss: 15.0    Val Loss: 30.0
Epoch 10: Train Loss: 8.0     Val Loss: 35.0    ← BAD: val going up!
Epoch 15: Train Loss: 4.0     Val Loss: 45.0    ← model is memorizing, not learning
```

**What this means:** The model is memorizing the training data instead of learning general patterns. It will sound bad on new voices.

**What to do:** The best checkpoint is from around epoch 5 (lowest val loss). The `best_model.pt` file is automatically saved from the epoch with the lowest val loss, so it should be fine even if this happens.

### 2. Loss Explodes (Jumps to Very High Numbers)

```
Epoch 3: Train Loss: 20.0
Epoch 4: Train Loss: 850.0    ← something broke
```

**What to do:** Training has likely diverged. You'd need to restart with a lower learning rate.

### 3. Loss Stuck / Not Decreasing

```
Epoch 0:  Train Loss: 35.0
Epoch 5:  Train Loss: 34.8
Epoch 10: Train Loss: 34.5     ← barely moving
```

**What this means:** The model isn't learning effectively. Could be a learning rate issue or data problem.

### 4. D_Loss Goes to 0

If the discriminator loss drops to nearly 0 and stays there, it means the discriminator is "winning" too easily. The generator can't learn because the feedback is too harsh. This is a known GAN problem called "mode collapse."

---

## Can I Just Leave It Running?

**Yes!** The training is designed to run and finish on its own. Here's what happens automatically:

1. Runs for **20 reporting epochs**
2. Each epoch: **7000 training steps + 200 validation steps**
3. **Saves a checkpoint after every epoch** (`checkpoint_epoch_0_step_0.pt`, etc.)
4. **Saves the best model** (`best_model.pt`) whenever val loss improves
5. **Stops automatically** when all 20 epochs are done
6. Returns the full result as JSON

Important: in the new setup, epochs are **step-based progress windows**, not literal "one pass through every sample in the dataset". The train stream now continues across epochs instead of restarting from the first samples each time.

**You do NOT need to manually stop it.** Just check on it once in a while.

### Suggested Check Schedule

| When | What to look for |
|---|---|
| After 30 min | Is it still `IN_PROGRESS`? (confirms no early crash) |
| After 2-3 hours | Still running? Good. First few epochs may be done. |
| Every few hours | Quick status check. |
| When `COMPLETED` | Read the JSON output. Check losses and checkpoints. |

---

## What If It Fails?

### `FAILED` Status

Run the check script and look at the error. Common reasons:
- **Out of memory (OOM):** GPU ran out of VRAM. Would need smaller batch size.
- **Dataset errors:** HuggingFace servers were flaky. Can just rerun.
- **Timeout:** Ran over the 24h limit. Checkpoints up to that point are still saved on the worker.

### How to Restart

Just run the launch command again:

```bash
.venv/bin/python scripts/flash_full_train.py
```

This deploys a fresh job. Previous checkpoints from a failed run are on the worker's temp storage (not persistent across workers).

---

## Understanding Checkpoints

A checkpoint is a snapshot of the model at a specific point in training. Think of it as a save game.

| File | What It Is |
|---|---|
| `checkpoint_epoch_0_step_0.pt` | Model state after epoch 0 |
| `checkpoint_epoch_1_step_0.pt` | Model state after epoch 1 |
| ... | One per epoch |
| `best_model.pt` | Copy of whichever epoch had the lowest val loss |

**`best_model.pt` is the one you want to use.** It's automatically the best-performing version.

Each checkpoint is ~512 MB (the full model weights).

---

## Training Config Summary

These are the settings being used (from `configs/mullivc_runpod_production.yaml`):

| Setting | Value | Meaning |
|---|---|---|
| `num_epochs` | 20 | Number of reporting epochs |
| `steps_per_epoch` | 7000 | Training batches per epoch |
| `validation_steps` | 200 | Validation batches per epoch |
| `batch_size` | 2 | Samples processed at once (limited by GPU memory) |
| `learning_rate` | 0.0001 | How big each learning step is |
| `save_interval` | 1 | Save checkpoint every N epochs |
| `max_validation_samples` | 2048 | Size of the stable validation slice |
| `max_train_samples` | None | No hidden train sample cap |

### Datasets

- **LibriTTS** (English speech, `train.clean.360` split) — main dataset
- **Fongbe Speech** (Fongbe language, male + female) — multilingual component

Both are streamed from HuggingFace (not downloaded in full).

---

## Quick Reference

```bash
# Check job status
.venv/bin/python scripts/check_job.py nivp8gwhwplup3 e3e5eecd-3494-483d-8c28-32749efd5eb0-e2

# On the worker terminal: show latest structured progress
python /tmp/MulliVC/scripts/monitor_training_progress.py /tmp/MulliVC/flash_runs/577788cf

# On the worker terminal: follow the human-readable log
tail -f /tmp/MulliVC/flash_runs/577788cf/logs/training.log

# Launch a new training run (if needed)
.venv/bin/python scripts/flash_full_train.py

# Launch with custom settings
.venv/bin/python scripts/flash_full_train.py --epochs 10 --batch-size 4

# The polling terminal also shows live status updates.
# Look for the terminal running flash_full_train.py.
```

---

## Glossary

| Term | Simple Explanation |
|---|---|
| **Epoch** | In this setup, a fixed window of training steps used for reporting and checkpointing |
| **Step/Batch** | Processing one small group of samples |
| **Loss** | How wrong the model is (lower = better) |
| **Train Loss** | Error on training data |
| **Val Loss** | Error on unseen data (the important one) |
| **GAN** | Two-network setup: generator makes audio, discriminator judges it |
| **Overfitting** | Model memorizes training data instead of learning patterns |
| **Checkpoint** | Saved snapshot of the model you can load later |
| **Convergence** | When loss stops meaningfully decreasing — training is "done" |
| **Learning Rate** | Size of each adjustment step (too big = unstable, too small = slow) |
| **Batch Size** | How many samples the model sees at once (limited by GPU memory) |
