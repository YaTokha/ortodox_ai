#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate

export BASE_MODEL_NAME="${BASE_MODEL_NAME:-ai-forever/rugpt3small_based_on_gpt2}"
export LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-outputs/lora-orthodox-gpt2}"
export APP_ENV="${APP_ENV:-prod}"

uvicorn app.main:app --host 0.0.0.0 --port 8000
