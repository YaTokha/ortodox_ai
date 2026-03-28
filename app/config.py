from functools import lru_cache
import os
from pathlib import Path

from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _HAS_PYDANTIC_SETTINGS = True
except ImportError:
    BaseSettings = BaseModel  # type: ignore[assignment]
    SettingsConfigDict = dict  # type: ignore[assignment]
    _HAS_PYDANTIC_SETTINGS = False


class Settings(BaseSettings):
    if _HAS_PYDANTIC_SETTINGS:
        model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_env: str = Field(default="dev", alias="APP_ENV")

    base_model_name: str = Field(default="ai-forever/rugpt3small_based_on_gpt2", alias="BASE_MODEL_NAME")
    lora_adapter_path: str = Field(default="", alias="LORA_ADAPTER_PATH")
    use_gpu_if_available: bool = Field(default=True, alias="USE_GPU_IF_AVAILABLE")
    max_input_chars: int = Field(default=12000, alias="MAX_INPUT_CHARS")
    disable_model: bool = Field(default=False, alias="DISABLE_MODEL")

    corpus_path: str = Field(default="data/processed/corpus.jsonl", alias="CORPUS_PATH")
    top_k_retrieval: int = Field(default=4, alias="TOP_K_RETRIEVAL")

    def corpus_abspath(self) -> Path:
        return (Path.cwd() / self.corpus_path).resolve()


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _load_dotenv_if_possible() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(".env")
    except Exception:
        pass


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    if _HAS_PYDANTIC_SETTINGS:
        return Settings()

    _load_dotenv_if_possible()
    # Fallback-режим: без pydantic-settings читаем переменные окружения вручную.
    return Settings(
        APP_HOST=os.getenv("APP_HOST", "0.0.0.0"),
        APP_PORT=int(os.getenv("APP_PORT", "8000")),
        APP_ENV=os.getenv("APP_ENV", "dev"),
        BASE_MODEL_NAME=os.getenv("BASE_MODEL_NAME", "ai-forever/rugpt3small_based_on_gpt2"),
        LORA_ADAPTER_PATH=os.getenv("LORA_ADAPTER_PATH", ""),
        USE_GPU_IF_AVAILABLE=_env_bool("USE_GPU_IF_AVAILABLE", True),
        MAX_INPUT_CHARS=int(os.getenv("MAX_INPUT_CHARS", "12000")),
        DISABLE_MODEL=_env_bool("DISABLE_MODEL", False),
        CORPUS_PATH=os.getenv("CORPUS_PATH", "data/processed/corpus.jsonl"),
        TOP_K_RETRIEVAL=int(os.getenv("TOP_K_RETRIEVAL", "4")),
    )
