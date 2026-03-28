#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-$HOME/ortodox_ai}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

SUDO=""
if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  if command -v sudo >/dev/null 2>&1; then
    SUDO="sudo"
  else
    echo "[ERROR] Требуется sudo для установки системных пакетов."
    exit 1
  fi
fi

$SUDO apt-get update
$SUDO apt-get install -y git rsync python3-venv python3-pip build-essential

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

if [[ ! -d .venv ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip wheel

# CUDA-версия PyTorch ставится отдельно (если есть GPU)
if command -v nvidia-smi >/dev/null 2>&1; then
  python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu121
else
  python -m pip install --upgrade torch
fi

python -m pip install -r requirements.txt

echo "[OK] Server setup completed in $PROJECT_DIR"
