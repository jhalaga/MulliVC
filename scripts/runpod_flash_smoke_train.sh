#!/usr/bin/env bash

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/jhalaga/MulliVC.git}"
CHECKOUT_REF="${CHECKOUT_REF:-ca3d4f76990036ddb729734cde491d516097952c}"
WORKDIR="${WORKDIR:-$HOME/MulliVC}"
PYTHON_VERSION="${PYTHON_VERSION:-3.13}"

echo "[1/6] Ensuring uv is installed"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "[2/6] Cloning or updating repository"
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone "$REPO_URL" "$WORKDIR"
fi

cd "$WORKDIR"
git fetch origin
git checkout "$CHECKOUT_REF"

echo "[3/6] Installing Python $PYTHON_VERSION"
uv python install "$PYTHON_VERSION"

echo "[4/6] Creating virtual environment"
uv venv --python "$PYTHON_VERSION" .venv
source .venv/bin/activate

echo "[5/6] Installing requirements"
uv pip install -r requirements.txt

echo "[6/6] Starting smoke training"
python train.py \
  --config configs/mullivc_runpod.yaml \
  --epochs 1 \
  --max-train-samples 256 \
  --max-val-samples 64 \
  --steps-per-epoch 10 \
  --validation-steps 2 \
  --disable-wandb