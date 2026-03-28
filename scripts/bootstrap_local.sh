#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp -n .env.example .env || true

python -m train.prepare_dataset --raw-root data/raw
python main.py
