#!/usr/bin/env bash
set -euo pipefail

BASE_MODEL="${BASE_MODEL:-ai-forever/rugpt3small_based_on_gpt2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/lora-orthodox-gpt2}"
TRAIN_FILE="${TRAIN_FILE:-data/processed/train.jsonl}"
VALID_FILE="${VALID_FILE:-data/processed/valid.jsonl}"
TEST_FILE="${TEST_FILE:-data/processed/test.jsonl}"
MAX_LENGTH="${MAX_LENGTH:-256}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-370}"

source .venv/bin/activate

if [[ ! -f "$HOME/.cache/huggingface/accelerate/default_config.yaml" ]]; then
  accelerate config default
fi

python -m train.prepare_dataset \
  --raw-root data/raw \
  --out-corpus data/processed/corpus.jsonl \
  --out-train "$TRAIN_FILE" \
  --out-valid "$VALID_FILE" \
  --out-test "$TEST_FILE"

accelerate launch train/train_lora.py \
  --train-file "$TRAIN_FILE" \
  --valid-file "$VALID_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --base-model "$BASE_MODEL" \
  --max-length "$MAX_LENGTH" \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --num-epochs "$NUM_EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --logging-steps "$LOGGING_STEPS" \
  --save-steps "$SAVE_STEPS"

python -m train.eval_perplexity \
  --test-file "$TEST_FILE" \
  --base-model "$BASE_MODEL" \
  --adapter-path "$OUTPUT_DIR"

echo "[OK] Training complete. Adapter path: $OUTPUT_DIR"
