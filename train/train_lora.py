"""Дообучение GPT-2 / ruGPT в формате LoRA (PEFT)."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import inspect
from typing import Dict, List

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


@dataclass
class TrainConfig:
    train_file: str
    valid_file: str
    output_dir: str
    base_model: str
    max_length: int
    per_device_batch_size: int
    grad_accum_steps: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    logging_steps: int
    save_steps: int
    fp16: bool
    bf16: bool


def guess_target_modules(model: torch.nn.Module) -> List[str]:
    # Универсальная эвристика: вытаскиваем имена часто используемых линейных слоёв.
    candidates = {"c_attn", "c_proj", "c_fc", "q_proj", "k_proj", "v_proj", "o_proj"}
    found = set()
    for name, _ in model.named_modules():
        last = name.split(".")[-1]
        if last in candidates:
            found.add(last)
    return sorted(found) or ["c_attn"]


def format_example(row: Dict[str, str]) -> str:
    return (
        f"[SOURCE={row.get('source_type', 'unknown')}] "
        f"[AUTHOR={row.get('author', 'unknown')}] "
        f"[TITLE={row.get('title', '')}] "
        f"[REF={row.get('reference', '')}]\n"
        f"{row.get('text', '').strip()}"
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", default="data/processed/train.jsonl")
    parser.add_argument("--valid-file", default="data/processed/valid.jsonl")
    parser.add_argument("--output-dir", default="outputs/lora-orthodox-gpt2")
    parser.add_argument("--base-model", default="gpt2")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    return TrainConfig(
        train_file=args.train_file,
        valid_file=args.valid_file,
        output_dir=args.output_dir,
        base_model=args.base_model,
        max_length=args.max_length,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        fp16=args.fp16,
        bf16=args.bf16,
    )


def main() -> None:
    cfg = parse_args()
    os.makedirs(cfg.output_dir, exist_ok=True)

    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()
    device_name = "mps" if use_mps else "cuda" if use_cuda else "cpu"
    if use_mps:
        # На Apple Silicon часть операций может падать без fallback.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print("Training device:", device_name)

    print("Loading tokenizer/model:", cfg.base_model)
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    target_modules = guess_target_modules(model)
    print("LoRA target modules:", target_modules)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    data_files = {"train": cfg.train_file, "validation": cfg.valid_file}
    ds = load_dataset("json", data_files=data_files)
    has_eval = "validation" in ds and len(ds["validation"]) > 0

    def preprocess(batch):
        texts = [format_example(row) for row in batch]
        tokens = tokenizer(
            texts,
            truncation=True,
            max_length=cfg.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    # Токенизация прямо на datasets для удобной работы с большими корпусами.
    tokenized_ds = ds.map(
        lambda examples: preprocess(
            [
                {
                    "source_type": s,
                    "author": a,
                    "title": t,
                    "reference": r,
                    "text": txt,
                }
                for s, a, t, r, txt in zip(
                    examples.get("source_type", []),
                    examples.get("author", []),
                    examples.get("title", []),
                    examples.get("reference", []),
                    examples.get("text", []),
                )
            ]
        ),
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    evaluation_strategy = "steps" if has_eval else "no"
    training_args_kwargs = dict(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_batch_size,
        per_device_eval_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.save_steps,
        save_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=has_eval,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        use_mps_device=use_mps,
        no_cuda=not (use_cuda or use_mps),
        dataloader_pin_memory=False,
        report_to="none",
    )
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" in ta_params:
        training_args_kwargs["eval_strategy"] = evaluation_strategy
    else:
        training_args_kwargs["evaluation_strategy"] = evaluation_strategy

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"] if has_eval else None,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    metrics = trainer.evaluate() if has_eval else {}
    metrics_path = Path(cfg.output_dir) / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Training finished. Adapter saved to:", cfg.output_dir)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
