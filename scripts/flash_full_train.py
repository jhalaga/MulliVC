"""Run production-oriented MulliVC training on Runpod Flash."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex
import uuid

from runpod_flash import Endpoint, GpuGroup
from runpod_flash.core.resources.network_volume import NetworkVolume, DataCenter
import runpod.endpoint.runner as _rp_runner

# Increase HTTP read timeout from 10s to 60s to survive Runpod API slowness.
_rp_runner.RunPodClient.get = lambda self, endpoint, timeout=60: _rp_runner.RunPodClient._request(self, "GET", endpoint, timeout=timeout)
_rp_runner.RunPodClient.post = lambda self, endpoint, data, timeout=60: _rp_runner.RunPodClient._request(self, "POST", endpoint, data, timeout)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/mullivc_runpod_production.yaml"
DEFAULT_REQUIREMENTS = [
    line.strip()
    for line in (PROJECT_ROOT / "requirements.txt").read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
TRAINING_REQUIREMENTS = [
    requirement
    for requirement in DEFAULT_REQUIREMENTS
    if not requirement.startswith("openai-whisper")
    and not requirement.startswith("matplotlib")
    and not requirement.startswith("torch")
]
if not any(requirement.startswith("soundfile") for requirement in TRAINING_REQUIREMENTS):
    TRAINING_REQUIREMENTS.append("soundfile")
FORWARDED_ENV = {
    key: value
    for key, value in {
        "HF_TOKEN": os.getenv("HF_TOKEN"),
    }.items()
    if value
}


def _synced_repo_files() -> dict[str, str]:
    return {
        "train.py": (PROJECT_ROOT / "train.py").read_text(),
        "models/content_encoder.py": (PROJECT_ROOT / "models" / "content_encoder.py").read_text(),
        "models/discriminator.py": (PROJECT_ROOT / "models" / "discriminator.py").read_text(),
        "models/losses.py": (PROJECT_ROOT / "models" / "losses.py").read_text(),
        "models/mullivc.py": (PROJECT_ROOT / "models" / "mullivc.py").read_text(),
        "utils/data_utils.py": (PROJECT_ROOT / "utils" / "data_utils.py").read_text(),
        "utils/model_utils.py": (PROJECT_ROOT / "utils" / "model_utils.py").read_text(),
        "configs/mullivc_runpod_production.yaml": (PROJECT_ROOT / "configs" / "mullivc_runpod_production.yaml").read_text(),
        "scripts/monitor_training_progress.py": (PROJECT_ROOT / "scripts" / "monitor_training_progress.py").read_text(),
    }


@Endpoint(
    name="mullivc-full-train-v13",
    # Require >=80 GB VRAM. L40S (48 GB) OOM'd on the attention T^2 matrix
    # at batch_size=16 in v12.
    gpu=[
        GpuGroup.HOPPER_141,     # H200 141 GB
        GpuGroup.BLACKWELL_180,  # B200 180 GB
        GpuGroup.BLACKWELL_96,   # RTX PRO 6000 Blackwell 96 GB
        GpuGroup.AMPERE_80,      # A100 80 GB
        GpuGroup.ADA_80_PRO,     # L40S PRO 80 GB tier
    ],
    # Keep 1 worker permanently alive so the job is not terminated when the
    # platform scales down idle flex workers mid-run.
    workers=(1, 1),
    idle_timeout=900,
    dependencies=TRAINING_REQUIREMENTS,
    env=FORWARDED_ENV,
    execution_timeout_ms=604_800_000,
    volume=NetworkVolume(id="n9r4ol0ioh", dataCenterId=DataCenter.US_NC_1),
    datacenter=DataCenter.US_NC_1,
)
async def run_mullivc_full_train(request: dict | None = None) -> dict:
    """Execute a production-oriented MulliVC training run on a Flash GPU worker."""
    import json
    import os
    from pathlib import Path
    import shlex
    import shutil
    import subprocess
    import sys
    import time
    import traceback
    import uuid

    import yaml

    # Top-level try/except so we always return a useful error dict instead of
    # silently crashing the Flash worker.
    request = dict(request or {})
    synced_repo_files = request.pop("synced_repo_files", None)
    if not synced_repo_files:
        raise RuntimeError("synced_repo_files payload missing from request")
    default_config = "configs/mullivc_runpod_production.yaml"

    def _tail_lines(path: Path, limit: int = 120) -> list[str]:
        if not path.exists():
            return []
        return path.read_text(errors="replace").splitlines()[-limit:]

    repo_url = "https://github.com/jhalaga/MulliVC.git"
    clone_dir = Path("/tmp/MulliVC")
    if not (clone_dir / "train.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, str(clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    for relative_path, content in synced_repo_files.items():
        target_path = clone_dir / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(content)

    torch_marker = Path("/tmp/.torch_upgraded")
    if not torch_marker.exists():
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu124",
                "--quiet",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        torch_marker.touch()

    project_root = clone_dir
    config_path = project_root / request.get("config", default_config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_id = request.get("run_id") or uuid.uuid4().hex[:8]
    tmp_output_root = project_root / "flash_runs" / run_id
    if Path("/runpod-volume").exists():
        output_root = Path("/runpod-volume") / "mullivc_flash_runs" / run_id
    else:
        output_root = tmp_output_root

    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    eval_dir = output_root / "evaluation"
    pretrained_dir = output_root / "pretrained_models"
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    pretrained_dir.mkdir(parents=True, exist_ok=True)

    tmp_output_root.parent.mkdir(parents=True, exist_ok=True)
    if tmp_output_root != output_root:
        if tmp_output_root.is_symlink() or tmp_output_root.is_file():
            tmp_output_root.unlink()
        elif tmp_output_root.exists():
            shutil.rmtree(tmp_output_root)
        tmp_output_root.symlink_to(output_root, target_is_directory=True)

    with config_path.open("r") as file_handle:
        config = yaml.safe_load(file_handle)

    paths = config.setdefault("paths", {})
    paths["checkpoint_dir"] = str(checkpoint_dir)
    paths["log_dir"] = str(log_dir)
    paths["eval_dir"] = str(eval_dir)
    paths["pretrained_dir"] = str(pretrained_dir)

    data_config = config.setdefault("data", {})
    data_config["use_streaming"] = True
    data_config["num_workers"] = 0

    flash_config_path = output_root / "flash_config.yaml"
    with flash_config_path.open("w") as file_handle:
        yaml.safe_dump(config, file_handle, sort_keys=False)

    command = [
        sys.executable,
        "train.py",
        "--config",
        str(flash_config_path),
        "--disable-wandb",
    ]

    if request.get("epochs") is not None:
        command.extend(["--epochs", str(request["epochs"])])
    if request.get("batch_size") is not None:
        command.extend(["--batch-size", str(request["batch_size"])])
    if request.get("num_workers") is not None:
        command.extend(["--num-workers", str(request["num_workers"])])
    if request.get("max_train_samples") is not None:
        command.extend(["--max-train-samples", str(request["max_train_samples"])])
    if request.get("max_val_samples") is not None:
        command.extend(["--max-val-samples", str(request["max_val_samples"])])
    if request.get("steps_per_epoch") is not None:
        command.extend(["--steps-per-epoch", str(request["steps_per_epoch"])])
    if request.get("validation_steps") is not None:
        command.extend(["--validation-steps", str(request["validation_steps"])])

    # Resume from a previous checkpoint if specified
    resume_path = request.get("resume")
    if resume_path:
        resolved = project_root / resume_path if not Path(resume_path).is_absolute() else Path(resume_path)
        if resolved.exists():
            command.extend(["--resume", str(resolved)])
            print(f"[launcher] Resuming from {resolved}")
        else:
            print(f"[launcher] WARNING: resume path not found: {resolved}")

    timeout_seconds = int(request.get("timeout_seconds", 518_400))
    training_log_path = log_dir / "training.log"
    progress_path = log_dir / "progress.json"
    metrics_path = log_dir / "metrics.jsonl"
    metadata_path = log_dir / "run_metadata.json"

    metadata = {
        "run_id": run_id,
        "config": str(config_path),
        "flash_config": str(flash_config_path),
        "output_root": str(output_root),
        "monitor_root": str(tmp_output_root),
        "training_log_path": str(training_log_path),
        "progress_path": str(progress_path),
        "metrics_path": str(metrics_path),
        "command": " ".join(shlex.quote(part) for part in command),
        "started_at": time.time(),
        "timeout_seconds": timeout_seconds,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))

    started_at = time.time()
    timed_out = False

    with training_log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        log_handle.write(f"[launcher] run_id={run_id}\n")
        log_handle.write(f"[launcher] output_root={output_root}\n")
        log_handle.write(f"[launcher] monitor_root={tmp_output_root}\n")
        log_handle.write(f"[launcher] progress_path={progress_path}\n")
        log_handle.write(f"[launcher] metrics_path={metrics_path}\n")
        log_handle.write(f"[launcher] command={' '.join(shlex.quote(part) for part in command)}\n")

        process = subprocess.Popen(
            command,
            cwd=project_root,
            env={
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            returncode = process.wait()
            log_handle.write(f"[launcher] training timed out after {timeout_seconds}s\n")

    duration_seconds = round(time.time() - started_at, 2)
    checkpoint_files = []
    if checkpoint_dir.exists():
        checkpoint_files = [
            {
                "name": file_path.name,
                "size_bytes": file_path.stat().st_size,
            }
            for file_path in sorted(checkpoint_dir.glob("*.pt"))
        ]

    progress_snapshot = None
    if progress_path.exists():
        progress_snapshot = json.loads(progress_path.read_text())

    return {
        "ok": (not timed_out) and returncode == 0,
        "timed_out": timed_out,
        "returncode": returncode,
        "duration_seconds": duration_seconds,
        "command": " ".join(shlex.quote(part) for part in command),
        "run_id": run_id,
        "output_root": str(output_root),
        "monitor_root": str(tmp_output_root),
        "training_log_path": str(training_log_path),
        "progress_path": str(progress_path),
        "metrics_path": str(metrics_path),
        "checkpoint_files": checkpoint_files,
        "progress_snapshot": progress_snapshot,
        "stdout_tail": _tail_lines(training_log_path),
        "stderr_tail": [],
    }


def _build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "config": args.config,
        "timeout_seconds": args.timeout_seconds,
        "run_id": args.run_id or uuid.uuid4().hex[:8],
        "synced_repo_files": _synced_repo_files(),
    }
    if args.epochs is not None:
        payload["epochs"] = args.epochs
    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size
    if args.num_workers is not None:
        payload["num_workers"] = args.num_workers
    if args.max_train_samples is not None:
        payload["max_train_samples"] = args.max_train_samples
    if args.max_val_samples is not None:
        payload["max_val_samples"] = args.max_val_samples
    if args.steps_per_epoch is not None:
        payload["steps_per_epoch"] = args.steps_per_epoch
    if args.validation_steps is not None:
        payload["validation_steps"] = args.validation_steps
    if args.resume is not None:
        payload["resume"] = args.resume
    return payload


async def _main_async(args: argparse.Namespace):
    import base64
    import time

    import cloudpickle
    import requests as req
    from runpod_flash.core.credentials import get_api_key
    from runpod_flash.core.resources import ResourceManager
    from runpod_flash.stubs.live_serverless import LiveServerlessStub

    remote_cfg = run_mullivc_full_train.__remote_config__
    resource_config = remote_cfg["resource_config"]
    dependencies = remote_cfg.get("dependencies")
    system_deps = remote_cfg.get("system_dependencies")

    print("[flash] Deploying / reusing endpoint ...")
    resource_manager = ResourceManager()
    server = await resource_manager.get_or_deploy_resource(resource_config)
    endpoint_id = server.id
    if not endpoint_id:
        raise RuntimeError("Endpoint was not deployed (no id)")
    print(f"[flash] Endpoint ready: {endpoint_id}")

    print("[flash] Serializing function ...")
    stub = LiveServerlessStub(server)
    original_func = run_mullivc_full_train.__wrapped__
    payload = _build_payload(args)
    run_id = payload["run_id"]
    request_obj = await stub.prepare_request(
        original_func,
        dependencies,
        system_deps,
        True,
        payload,
    )
    raw_payload = request_obj.model_dump(exclude_none=True)

    job = await asyncio.to_thread(server.endpoint.run, request_input=raw_payload)
    job_id = job.job_id
    print(f"[flash] Job submitted: {job_id}")
    print(f"[flash] Endpoint ID: {endpoint_id}")
    print(f"[flash] Run ID: {run_id}")
    print("[flash]")
    print("[flash] === MONITOR THIS JOB ===")
    print("[flash] You can safely close this terminal.")
    print("[flash] Local status check:")
    print(f"[flash]   .venv/bin/python scripts/check_job.py {endpoint_id} {job_id}")
    print("[flash] Worker-side monitoring paths:")
    print(f"[flash]   /tmp/MulliVC/flash_runs/{run_id}/logs/training.log")
    print(f"[flash]   /tmp/MulliVC/flash_runs/{run_id}/logs/progress.json")
    print(f"[flash]   /tmp/MulliVC/flash_runs/{run_id}/logs/metrics.jsonl")
    print("[flash] Worker-side commands:")
    print(f"[flash]   python /tmp/MulliVC/scripts/monitor_training_progress.py /tmp/MulliVC/flash_runs/{run_id}")
    print(f"[flash]   tail -f /tmp/MulliVC/flash_runs/{run_id}/logs/training.log")
    print("[flash] ===========================")
    print("[flash]")

    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    poll_interval = 30
    start = time.time()
    last_status = None
    while True:
        await asyncio.sleep(poll_interval)
        elapsed = time.time() - start
        try:
            resp = req.get(status_url, headers=headers, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"[flash] [{elapsed:.0f}s] Poll error (will retry): {exc}")
            continue

        status = data.get("status", "UNKNOWN")
        if status != last_status:
            print(f"[flash] [{elapsed:.0f}s] Status: {status}")
            last_status = status
        elif int(elapsed) % 300 < poll_interval:
            hours = elapsed / 3600
            print(f"[flash] [{elapsed:.0f}s / {hours:.1f}h] Still {status} ...")

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break

    output = data.get("output", {})
    if status == "COMPLETED" and isinstance(output, dict) and output.get("result"):
        result = cloudpickle.loads(base64.b64decode(output["result"]))
    elif isinstance(output, dict) and output.get("json_result"):
        result = output["json_result"]
    else:
        result = {
            "ok": False,
            "status": status,
            "error": data.get("error") or str(output),
        }

    print(json.dumps(result, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        description="Launch production-oriented MulliVC training on Runpod Flash."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path inside the repo")
    parser.add_argument("--run-id", default=None, help="Optional explicit run identifier")
    parser.add_argument("--epochs", type=int, default=None, help="Override num_epochs from config")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit training samples")
    parser.add_argument("--max-val-samples", type=int, default=None, help="Limit validation samples")
    parser.add_argument("--steps-per-epoch", type=int, default=None, help="Override steps per epoch")
    parser.add_argument("--validation-steps", type=int, default=None, help="Override validation steps")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from (on worker)")
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=518_400,
        help="Worker-side timeout (default: 6 days)",
    )

    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()