#!/usr/bin/env bash
set -euo pipefail

# Пример:
# bash scripts/deploy_remote_train.sh root@SERVER_IP /root/ortodox_ai

REMOTE="${1:?Укажите remote, например root@SERVER_IP}"
REMOTE_DIR="${2:-~/ortodox_ai}"
BASE_MODEL="${BASE_MODEL:-ai-forever/rugpt3small_based_on_gpt2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora-orthodox-gpt2}"
MAX_LENGTH="${MAX_LENGTH:-256}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-370}"

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[1/3] Sync project -> $REMOTE:$REMOTE_DIR"
rsync -avz \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '.pytest_cache' \
  --exclude '.idea' \
  "$LOCAL_DIR/" "$REMOTE:$REMOTE_DIR/"

echo "[2/3] Setup env on remote"
ssh "$REMOTE" "cd $REMOTE_DIR && bash scripts/setup_server.sh $REMOTE_DIR"

echo "[3/3] Start training"
ssh "$REMOTE" "cd $REMOTE_DIR && \
BASE_MODEL='$BASE_MODEL' OUTPUT_DIR='$OUTPUT_DIR' \
MAX_LENGTH='$MAX_LENGTH' PER_DEVICE_BATCH_SIZE='$PER_DEVICE_BATCH_SIZE' \
GRAD_ACCUM_STEPS='$GRAD_ACCUM_STEPS' NUM_EPOCHS='$NUM_EPOCHS' \
LEARNING_RATE='$LEARNING_RATE' LOGGING_STEPS='$LOGGING_STEPS' SAVE_STEPS='$SAVE_STEPS' \
bash scripts/run_training.sh"

echo "[DONE] Training finished. Adapter: $OUTPUT_DIR"
