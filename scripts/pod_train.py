#!/usr/bin/env python3
"""Launch full MulliVC training on a standard Runpod pod (non-serverless).

Why a standard pod?
  - Serverless endpoints in US-NC-1 have intermittent capacity and will
    queue / scale down. A persistent pod is billed by the second and
    never scales below 1 -> no mid-run terminations.
  - We can SSH directly, attach the existing network volume (dataset +
    checkpoints), and use tmux to keep training alive across reconnects.

Flow:
  1. Ensure our local SSH pubkey is registered on the Runpod account.
  2. create_pod() in US-NC-1 attached to network volume n9r4ol0ioh,
     trying H200 -> H100 -> Blackwell 96 GB -> A100 80 GB.
  3. Wait until SSH is reachable.
  4. rsync the repo to /workspace/MulliVC on the pod.
  5. Kick off training inside a detached tmux session on the pod,
     streaming logs to /runpod-volume/pod_logs/<run_id>/training.log.
  6. Print connection info.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests
from runpod_flash.core.credentials import get_api_key
import runpod

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_PUBKEY = Path.home() / ".ssh" / "id_ed25519.pub"
LOCAL_PRIVKEY = Path.home() / ".ssh" / "id_ed25519"
VOLUME_ID = "n9r4ol0ioh"           # jhdata, US-NC-1
DATACENTER = "US-NC-1"
# Try most capable first. Cost-per-hour quoted from US-NC-1 on 2026-04-18.
GPU_PREFERENCE = [
    ("NVIDIA H200", 3.59),
    ("NVIDIA H100 80GB HBM3", 2.99),
    ("NVIDIA H100 PCIe", 2.39),
    ("NVIDIA H100 NVL", 2.79),
    ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 1.69),
    ("NVIDIA A100-SXM4-80GB", 1.89),
    ("NVIDIA A100 80GB PCIe", 1.64),
]
# PyTorch 2.4 / CUDA 12.4 / py3.11 -- compatible with H100/H200/A100.
# (RTX PRO 6000 Blackwell sm_120 would want CUDA 12.8; it's a last-resort
# fallback. If that card is the only one available, rerun with
# --image runpod/pytorch:2.6.0-py3.11-cuda12.8.1-devel-ubuntu22.04.)
DEFAULT_IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


def _gql(api_key: str, query: str, variables: dict | None = None) -> dict:
    r = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"query": query, "variables": variables or {}},
        timeout=30,
    )
    r.raise_for_status()
    payload = r.json()
    if "errors" in payload:
        raise RuntimeError(f"GraphQL error: {payload['errors']}")
    return payload["data"]


def ensure_ssh_key_registered(api_key: str) -> None:
    if not LOCAL_PUBKEY.exists():
        sys.exit(f"Missing {LOCAL_PUBKEY}. Generate with: ssh-keygen -t ed25519")
    local_key = LOCAL_PUBKEY.read_text().strip()
    data = _gql(api_key, "{ myself { pubKey } }")
    current = (data["myself"].get("pubKey") or "").strip()
    if local_key in current:
        return
    new_value = (current + "\n" + local_key).strip() + "\n"
    mut = """
    mutation($input: UpdateUserSettingsInput!) {
      updateUserSettings(input: $input) { id }
    }
    """
    _gql(api_key, mut, {"input": {"pubKey": new_value}})
    print("[ssh] Registered local public key with Runpod account.")


def create_pod_with_fallback(run_id: str, image: str, gpu_types: list[str]) -> tuple[str, str]:
    last_err = None
    for gpu_type in gpu_types:
        try:
            print(f"[pod] Trying GPU type: {gpu_type}")
            pod = runpod.create_pod(
                name=f"mullivc-train-{run_id}",
                image_name=image,
                gpu_type_id=gpu_type,
                gpu_count=1,
                cloud_type="SECURE",
                data_center_id=DATACENTER,
                support_public_ip=True,
                start_ssh=True,
                volume_in_gb=0,
                container_disk_in_gb=80,
                min_vcpu_count=8,
                min_memory_in_gb=32,
                ports="22/tcp",
                volume_mount_path="/runpod-volume",
                network_volume_id=VOLUME_ID,
                env={
                    "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                    "POD_RUN_ID": run_id,
                    "PYTHONUNBUFFERED": "1",
                },
            )
            print(f"[pod] Created: id={pod['id']}  gpu={gpu_type}")
            return pod["id"], gpu_type
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)[:200]
            print(f"[pod]   unavailable ({msg})")
    raise RuntimeError(f"No GPU in preference list available in {DATACENTER}: {last_err}")


def wait_for_ssh(pod_id: str, timeout_s: int = 600) -> tuple[str, int]:
    """Poll until the pod has public SSH and the port is reachable."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        pod = runpod.get_pod(pod_id)
        runtime = (pod or {}).get("runtime") or {}
        ports = runtime.get("ports") or []
        for p in ports:
            if p.get("privatePort") == 22 and p.get("isIpPublic") and p.get("publicPort"):
                host = p["ip"]
                port = int(p["publicPort"])
                # probe
                probe = subprocess.run(
                    ["ssh",
                     "-o", "StrictHostKeyChecking=no",
                     "-o", "UserKnownHostsFile=/dev/null",
                     "-o", "ConnectTimeout=5",
                     "-o", "BatchMode=yes",
                     "-i", str(LOCAL_PRIVKEY),
                     "-p", str(port),
                     f"root@{host}",
                     "echo ready"],
                    capture_output=True, text=True, timeout=15,
                )
                if probe.returncode == 0 and "ready" in probe.stdout:
                    return host, port
        print(f"[pod] waiting for SSH... (status={pod.get('desiredStatus')}, ports={len(ports)})")
        time.sleep(10)
    raise TimeoutError(f"Pod {pod_id} SSH did not come up within {timeout_s}s")


def ssh_run(host: str, port: int, cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    full = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-i", str(LOCAL_PRIVKEY),
        "-p", str(port),
        f"root@{host}",
        cmd,
    ]
    return subprocess.run(full, check=check, text=True)


def rsync_repo(host: str, port: int, remote_dir: str) -> None:
    ssh_opts = (
        f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        f"-i {LOCAL_PRIVKEY} -p {port}"
    )
    ssh_run(host, port, f"mkdir -p {shlex.quote(remote_dir)}")
    subprocess.run(
        ["rsync", "-az", "--delete",
         "-e", ssh_opts,
         "--exclude=.git/",
         "--exclude=.venv/",
         "--exclude=__pycache__/",
         "--exclude=checkpoints/",
         "--exclude=logs/",
         "--exclude=data/",
         "--exclude=wandb/",
         "--exclude=*.pyc",
         f"{PROJECT_ROOT}/",
         f"root@{host}:{remote_dir}/"],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--config", default="configs/mullivc_runpod_production.yaml")
    parser.add_argument(
        "--gpu",
        nargs="*",
        default=None,
        help="Override GPU preference list (exact gpu_type_id values).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    api_key = get_api_key()
    runpod.api_key = api_key

    run_id = "pod_" + dt.datetime.now().strftime("%Y%m%d_%H%M")
    print(f"[run] id={run_id}")

    ensure_ssh_key_registered(api_key)

    gpu_types = args.gpu if args.gpu else [g[0] for g in GPU_PREFERENCE]
    if args.dry_run:
        print("[dry-run] Would create pod with GPUs:", gpu_types)
        return

    pod_id, gpu_type = create_pod_with_fallback(run_id, args.image, gpu_types)
    print(f"[pod] Waiting for SSH on pod {pod_id}...")
    try:
        host, port = wait_for_ssh(pod_id)
    except Exception:
        print(f"[pod] SSH not ready. To debug / terminate: runpod.terminate_pod('{pod_id}')")
        raise
    print(f"[ssh] root@{host}:{port}")

    remote_dir = "/workspace/MulliVC"
    print(f"[sync] rsync to {remote_dir}")
    rsync_repo(host, port, remote_dir)

    log_dir = f"/runpod-volume/pod_logs/{run_id}"
    ckpt_dir = f"/runpod-volume/checkpoints_{run_id}"
    # Bootstrap script that runs inside the pod: install deps, launch
    # training in a detached tmux session so it survives disconnects.
    hf_token = os.getenv("HF_TOKEN", "")
    bootstrap = f"""
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq >/dev/null && apt-get install -y -qq tmux rsync libsndfile1 ffmpeg >/dev/null
mkdir -p {shlex.quote(log_dir)} {shlex.quote(ckpt_dir)}
cd {shlex.quote(remote_dir)}
# Install Python deps (PyTorch already in base image -- skip torch line).
python -m pip install --upgrade pip >/dev/null
grep -v '^torch' requirements.txt | grep -v '^openai-whisper' | grep -v '^matplotlib' > /tmp/reqs.txt
python -m pip install --no-cache-dir -r /tmp/reqs.txt
# Launch in tmux, streaming output to both tmux and log file.
tmux kill-session -t train 2>/dev/null || true
tmux new-session -d -s train -- bash -lc '
  set -o pipefail;
  export HF_TOKEN={shlex.quote(hf_token)};
  export POD_RUN_ID={shlex.quote(run_id)};
  export PYTHONUNBUFFERED=1;
  cd {shlex.quote(remote_dir)};
  python -u train.py \\
    --config {shlex.quote(args.config)} \\
    --checkpoint_dir {shlex.quote(ckpt_dir)} \\
    --log_dir {shlex.quote(log_dir)} \\
    2>&1 | tee {shlex.quote(log_dir + "/training.log")}
'
echo "[bootstrap] tmux session 'train' started."
"""
    print("[run] Executing bootstrap on pod...")
    ssh_run(host, port, bootstrap)

    print()
    print("=" * 70)
    print(f"Pod ID:   {pod_id}")
    print(f"GPU:      {gpu_type}")
    print(f"SSH:      ssh -i {LOCAL_PRIVKEY} -p {port} root@{host}")
    print(f"Attach:   tmux attach -t train")
    print(f"Tail log: tail -f {log_dir}/training.log")
    print(f"Terminate when done:")
    print(f"  python -c \"import runpod; from runpod_flash.core.credentials import get_api_key; "
          f"runpod.api_key=get_api_key(); runpod.terminate_pod('{pod_id}')\"")
    print("=" * 70)


if __name__ == "__main__":
    main()
