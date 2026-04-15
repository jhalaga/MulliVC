"""Run a bounded MulliVC smoke training job on Runpod Flash."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import shlex

from runpod_flash import Endpoint, GpuGroup
import runpod.endpoint.runner as _rp_runner

# Increase HTTP read timeout from 10s to 60s to survive Runpod API slowness
_rp_runner.RunPodClient.get = lambda self, endpoint, timeout=60: _rp_runner.RunPodClient._request(self, "GET", endpoint, timeout=timeout)
_rp_runner.RunPodClient.post = lambda self, endpoint, data, timeout=60: _rp_runner.RunPodClient._request(self, "POST", endpoint, data, timeout)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/mullivc_runpod.yaml"
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


@Endpoint(
    name="mullivc-smoke-train-v11",
    gpu=GpuGroup.ADA_48_PRO,
    workers=(0, 1),
    idle_timeout=900,
    dependencies=TRAINING_REQUIREMENTS,
    env=FORWARDED_ENV,
    execution_timeout_ms=7_200_000,
)
async def run_mullivc_smoke_train(request: dict | None = None) -> dict:
    """Execute a bounded smoke training run on a Flash GPU worker."""
    import os
    from pathlib import Path
    import shlex
    import subprocess
    import sys
    import time
    import uuid

    import yaml

    request = request or {}

    # On the remote worker, project files aren't available (Flash only serializes the function).
    # Clone the repo to get train.py, configs, model code, etc.
    _repo_url = "https://github.com/jhalaga/MulliVC.git"
    _clone_dir = Path("/tmp/MulliVC")
    if not (_clone_dir / "train.py").exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", _repo_url, str(_clone_dir)],
            check=True,
            capture_output=True,
            text=True,
        )

    # FIX 1: Install torch + torchaudio built for CUDA 12.4 so they support
    # the current GPU (e.g. L40S sm_89) AND the worker's NVIDIA driver.
    # The Flash base image's torch lacks sm_89; a blanket --upgrade pulls
    # a cu129 build whose driver requirement exceeds the image driver (12.8).
    # Pinning cu124 ensures compatibility with the driver while adding sm_89.
    _torch_marker = Path("/tmp/.torch_upgraded")
    if not _torch_marker.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "torch", "torchaudio",
             "--index-url", "https://download.pytorch.org/whl/cu124",
             "--quiet"],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,
        )
        _torch_marker.touch()

    # FIX 1b: Patch the cloned repo's GPU detection to use the actual CUDA
    # allocation test instead of the strict arch-list check.  sm_89 (Ada)
    # works via PTX forward-compat even on sm_80-compiled torch builds.
    _model_utils = _clone_dir / "utils" / "model_utils.py"
    _mu_text = _model_utils.read_text()
    if "current_arch not in supported_arches" in _mu_text:
        _mu_text = _mu_text.replace(
            """        try:
            capability = torch.cuda.get_device_capability(0)
            current_arch = f"sm_{capability[0]}{capability[1]}"
            supported_arches = {
                arch for arch in torch.cuda.get_arch_list()
                if arch.startswith('sm_')
            }

            if supported_arches and current_arch not in supported_arches:
                device_name = torch.cuda.get_device_name(0)
                print(
                    f"GPU {device_name} ({current_arch}) is unsupported by the installed PyTorch build; using CPU instead."
                )
                return torch.device('cpu')

            torch.empty(1, device='cuda')
            return torch.device('cuda')""",
            """        try:
            torch.empty(1, device='cuda')
            return torch.device('cuda')""",
        )
        _model_utils.write_text(_mu_text)

    project_root = _clone_dir
    config_path = project_root / request.get("config", "configs/mullivc_runpod.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    run_id = request.get("run_id") or uuid.uuid4().hex[:8]
    if Path("/runpod-volume").exists():
        output_root = Path("/runpod-volume") / "mullivc_flash_runs" / run_id
    else:
        output_root = project_root / "flash_runs" / run_id

    checkpoint_dir = output_root / "checkpoints"
    log_dir = output_root / "logs"
    eval_dir = output_root / "evaluation"
    pretrained_dir = output_root / "pretrained_models"
    output_root.mkdir(parents=True, exist_ok=True)

    with config_path.open("r") as file_handle:
        config = yaml.safe_load(file_handle)

    paths = config.setdefault("paths", {})
    paths["checkpoint_dir"] = str(checkpoint_dir)
    paths["log_dir"] = str(log_dir)
    paths["eval_dir"] = str(eval_dir)
    paths["pretrained_dir"] = str(pretrained_dir)

    # FIX 2: Use streaming mode to avoid downloading the full dataset.
    # Non-streaming mode fails on workers that lack disk space or have
    # transient network issues while extracting multi-GB archives.
    data_config = config.setdefault("data", {})
    data_config["use_streaming"] = True

    flash_config_path = output_root / "flash_config.yaml"
    with flash_config_path.open("w") as file_handle:
        yaml.safe_dump(config, file_handle, sort_keys=False)

    command = [
        sys.executable,
        "train.py",
        "--config",
        str(flash_config_path),
        "--epochs",
        str(request.get("epochs", 1)),
        "--max-train-samples",
        str(request.get("max_train_samples", 256)),
        "--max-val-samples",
        str(request.get("max_val_samples", 64)),
        "--steps-per-epoch",
        str(request.get("steps_per_epoch", 10)),
        "--validation-steps",
        str(request.get("validation_steps", 2)),
        "--disable-wandb",
    ]

    if request.get("batch_size") is not None:
        command.extend(["--batch-size", str(request["batch_size"])])

    if request.get("num_workers") is not None:
        command.extend(["--num-workers", str(request["num_workers"])])

    timeout_seconds = int(request.get("timeout_seconds", 3600))
    started_at = time.time()

    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        result = exc
        timed_out = True

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

    if timed_out:
        stdout_text = result.stdout or ""
        stderr_text = result.stderr or ""
        return {
            "ok": False,
            "timed_out": True,
            "duration_seconds": duration_seconds,
            "timeout_seconds": timeout_seconds,
            "command": " ".join(shlex.quote(part) for part in command),
            "output_root": str(output_root),
            "checkpoint_files": checkpoint_files,
            "stdout_tail": stdout_text.splitlines()[-80:],
            "stderr_tail": stderr_text.splitlines()[-80:],
        }

    return {
        "ok": result.returncode == 0,
        "timed_out": False,
        "returncode": result.returncode,
        "duration_seconds": duration_seconds,
        "command": " ".join(shlex.quote(part) for part in command),
        "output_root": str(output_root),
        "checkpoint_files": checkpoint_files,
        "stdout_tail": result.stdout.splitlines()[-80:],
        "stderr_tail": result.stderr.splitlines()[-80:],
    }


def _build_payload(args: argparse.Namespace) -> dict:
    payload = {
        "config": args.config,
        "epochs": args.epochs,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "steps_per_epoch": args.steps_per_epoch,
        "validation_steps": args.validation_steps,
        "timeout_seconds": args.timeout_seconds,
    }

    if args.batch_size is not None:
        payload["batch_size"] = args.batch_size
    if args.num_workers is not None:
        payload["num_workers"] = args.num_workers

    return payload


async def _main_async(args: argparse.Namespace):
    # FIX 3: Bypass the Flash SDK's built-in polling/log-streaming loop
    # which hangs for 10+ minutes.  We replicate only deployment +
    # submission, then poll the public REST API ourselves.
    import base64
    import time

    import cloudpickle
    import requests as req
    from runpod_flash.core.credentials import get_api_key
    from runpod_flash.core.resources import ResourceManager
    from runpod_flash.stubs.live_serverless import LiveServerlessStub

    # --- Step 1: deploy / reuse the endpoint ---
    # __remote_config__ is set by the @Endpoint / remote() decorator.
    remote_cfg = run_mullivc_smoke_train.__remote_config__
    resource_config = remote_cfg["resource_config"]
    dependencies = remote_cfg.get("dependencies")
    system_deps = remote_cfg.get("system_dependencies")

    print("[flash] Deploying / reusing endpoint …")
    resource_manager = ResourceManager()
    server = await resource_manager.get_or_deploy_resource(resource_config)
    endpoint_id = server.id
    if not endpoint_id:
        raise RuntimeError("Endpoint was not deployed (no id)")
    print(f"[flash] Endpoint ready: {endpoint_id}")

    # --- Step 2: serialise the worker function + payload ---
    print("[flash] Serialising function …")
    stub = LiveServerlessStub(server)
    original_func = run_mullivc_smoke_train.__wrapped__
    payload = _build_payload(args)
    request_obj = await stub.prepare_request(
        original_func,
        dependencies,
        system_deps,
        True,            # accelerate_downloads (Endpoint default)
        payload,         # positional arg forwarded to the worker function
    )
    raw_payload = request_obj.model_dump(exclude_none=True)

    # --- Step 3: submit the job (quick – no polling) ---
    # server.endpoint is a runpod.Endpoint; its .run() returns a Job
    # object immediately after the POST to /run.
    job = await asyncio.to_thread(server.endpoint.run, request_input=raw_payload)
    job_id = job.job_id
    print(f"[flash] Job submitted: {job_id}")

    # --- Step 4: poll via REST API with clear progress messages ---
    api_key = get_api_key()
    headers = {"Authorization": f"Bearer {api_key}"}
    status_url = f"https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}"

    poll_interval = 10
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
        elif int(elapsed) % 60 < poll_interval:
            # periodic heartbeat so the user knows we're still alive
            print(f"[flash] [{elapsed:.0f}s] Still {status} …")

        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break

    # --- Step 5: decode the result ---
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
        description="Launch a bounded MulliVC smoke training run on Runpod Flash."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Config path inside the repo")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=256,
        help="Maximum number of training samples",
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=64,
        help="Maximum number of validation samples",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=10,
        help="Maximum number of training batches per epoch",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=2,
        help="Maximum number of validation batches",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch size override")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Optional DataLoader worker override",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=3600,
        help="Worker-side timeout for the training subprocess",
    )

    args = parser.parse_args()
    asyncio.run(_main_async(args))


if __name__ == "__main__":
    main()