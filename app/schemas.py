from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Citation(BaseModel):
    id: str
    source_type: str
    author: Optional[str] = None
    title: Optional[str] = None
    reference: Optional[str] = None
    excerpt: str
    score: float


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=20, max_length=12000)
    question: Optional[str] = Field(default=None, max_length=500)
    top_k_sources: int = Field(default=4, ge=1, le=10)


class AnalyzeResponse(BaseModel):
    analysis: str
    key_themes: List[str]
    citations: List[Citation]
    disclaimer: str


class GenerateRequest(BaseModel):
    prompt: Optional[str] = Field(default=None, min_length=10, max_length=12000)
    topic: Optional[str] = Field(default=None, min_length=3, max_length=300)
    bible_text: Optional[str] = Field(default=None, max_length=12000)
    occasion: Optional[str] = Field(default=None, max_length=200)
    audience: Optional[str] = Field(default="приход", max_length=200)
    style: str = Field(default="пастырский", max_length=100)

    max_new_tokens: int = Field(default=520, ge=120, le=900)
    temperature: float = Field(default=0.95, ge=0.1, le=1.5)
    top_p: float = Field(default=0.97, ge=0.2, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    top_k_sources: int = Field(default=5, ge=1, le=10)

    @model_validator(mode="after")
    def validate_prompt_or_topic(self):
        has_prompt = bool((self.prompt or "").strip())
        has_topic = bool((self.topic or "").strip())
        if not has_prompt and not has_topic:
            raise ValueError("Укажите либо «промт для генерации», либо «тему проповеди».")
        return self


class GenerateResponse(BaseModel):
    sermon: str
    outline: List[str]
    citations: List[Citation]
    model_name: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    base_model_name: str
    adapter_loaded: bool
