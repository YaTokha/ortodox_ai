"""Оценка perplexity на test.jsonl."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def format_example(row: dict) -> str:
    return (
        f"[SOURCE={row.get('source_type', 'unknown')}] [AUTHOR={row.get('author', 'unknown')}] "
        f"[TITLE={row.get('title', '')}] [REF={row.get('reference', '')}]\n"
        f"{row.get('text', '').strip()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--adapter-path", default="")
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    rows = [json.loads(x) for x in Path(args.test_file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        raise SystemExit("test.jsonl пуст")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    losses = []
    with torch.no_grad():
        for row in rows:
            text = format_example(row)
            tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_length)
            tokens = {k: v.to(device) for k, v in tokens.items()}
            out = model(**tokens, labels=tokens["input_ids"])
            losses.append(out.loss.item())

    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    print(f"avg_loss={avg_loss:.4f}")
    print(f"perplexity={ppl:.4f}")


if __name__ == "__main__":
    main()
