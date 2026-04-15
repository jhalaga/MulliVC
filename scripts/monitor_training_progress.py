"""Prints the latest structured training progress from a worker run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _resolve_log_dir(path_arg: str) -> Path:
    path = Path(path_arg).resolve()
    if path.is_file():
        return path.parent
    if path.name == "logs":
        return path
    return path / "logs"


def _read_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _tail_jsonl(path: Path, limit: int = 5):
    if not path.exists():
        return []
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [json.loads(line) for line in lines[-limit:]]


def main():
    parser = argparse.ArgumentParser(description="Inspect the latest worker-side training progress.")
    parser.add_argument("path", help="Run directory or logs directory")
    args = parser.parse_args()

    log_dir = _resolve_log_dir(args.path)
    progress = _read_json(log_dir / "progress.json")
    metrics = _tail_jsonl(log_dir / "metrics.jsonl")
    checkpoint_dir = log_dir.parent / "checkpoints"

    if progress is None:
        raise SystemExit(f"No progress.json found under {log_dir}")

    print(f"Status: {progress.get('status')}")
    print(f"Updated: {progress.get('updated_at')}")
    print(f"Epoch: {progress.get('current_epoch')}")
    print(f"Completed epochs: {progress.get('completed_epochs')}")
    print(f"Phase: {progress.get('phase')}")
    print(f"Current batch: {progress.get('current_batch')} / {progress.get('epoch_batches')}")
    print(f"Best val loss: {progress.get('best_val_loss')}")
    print(f"Best epoch: {progress.get('best_epoch')}")
    print(f"Latest checkpoint: {progress.get('latest_checkpoint')}")

    if progress.get('last_train_losses'):
        print("Last train losses:")
        for key, value in progress['last_train_losses'].items():
            print(f"  {key}: {value:.4f}")

    if progress.get('last_validation_losses'):
        print("Last validation losses:")
        for key, value in progress['last_validation_losses'].items():
            print(f"  {key}: {value:.4f}")

    if metrics:
        print("Recent metrics:")
        for record in metrics:
            print(f"  {record.get('timestamp')} | {record.get('kind')} | epoch={record.get('epoch')} batch={record.get('batch')}")

    if checkpoint_dir.exists():
        checkpoints = sorted(checkpoint_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
        if checkpoints:
            print("Latest checkpoints:")
            for checkpoint in checkpoints[:5]:
                print(f"  {checkpoint.name} ({checkpoint.stat().st_size / (1024 ** 2):.1f} MB)")


if __name__ == "__main__":
    main()