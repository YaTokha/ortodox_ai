from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import Settings


@dataclass
class GenerationResult:
    text: str
    model_name: str


class SermonGenerator:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Any = None
        self.tokenizer: Any = None
        self.device = "cpu"
        self.model_name = settings.base_model_name
        self.adapter_loaded = False

    def load(self) -> None:
        if self.settings.disable_model:
            return
        if self.model is not None and self.tokenizer is not None:
            return

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            # Если тяжелые ML-зависимости не установлены, сервис остается в fallback-режиме.
            return

        use_cuda = self.settings.use_gpu_if_available and torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"

        tokenizer = AutoTokenizer.from_pretrained(self.settings.base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.settings.base_model_name)

        if self.settings.lora_adapter_path:
            base_model = PeftModel.from_pretrained(base_model, self.settings.lora_adapter_path)
            self.adapter_loaded = True

        base_model.to(device)
        base_model.eval()

        self.device = device
        self.model = base_model
        self.tokenizer = tokenizer

    @property
    def loaded(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
    ) -> GenerationResult:
        if not self.loaded:
            self.load()

        if not self.loaded:
            fallback = (
                "Черновик проповеди (fallback-режим):\n"
                "1) Вступление: обозначьте тему и евангельский контекст.\n"
                "2) Толкование: раскройте смысл отрывка через святоотеческую традицию.\n"
                "3) Практика: предложите конкретные шаги духовной жизни для паствы.\n"
                "4) Завершение: призыв к молитве, покаянию и делам милосердия."
            )
            return GenerationResult(text=fallback, model_name=f"{self.settings.base_model_name} (fallback)")

        import torch

        model_ctx = int(
            getattr(self.model.config, "n_positions", None)
            or getattr(self.model.config, "max_position_embeddings", 1024)
            or 1024
        )
        # Всегда резервируем место под генерацию, чтобы не выйти за длину контекста.
        reserve_for_generation = min(max_new_tokens, 96)
        max_input_len = max(64, model_ctx - reserve_for_generation)

        prev_trunc_side = getattr(self.tokenizer, "truncation_side", "right")
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len)
        self.tokenizer.truncation_side = prev_trunc_side
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_len = int(inputs["input_ids"].shape[1])
        allowed_new_tokens = max(16, min(max_new_tokens, model_ctx - input_len - 1))

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=allowed_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Декодируем только сгенерированное продолжение, без входного промпта.
        generated_tokens = out[0][input_len:]
        generated = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return GenerationResult(text=generated, model_name=self.model_name)
