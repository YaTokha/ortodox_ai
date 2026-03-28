from pathlib import Path
from typing import Any, Dict, List, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, GenerateRequest, GenerateResponse, HealthResponse
from app.services.assistant_service import OrthodoxAssistantService

settings = get_settings()
assistant = OrthodoxAssistantService(settings)

app = FastAPI(
    title="Православный Интеллектуальный Ассистент",
    description="Прототип интеллектуального ассистента для анализа Священного Писания и генерации проповедей",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
templates_dir = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=static_dir), name="static")
templates = Jinja2Templates(directory=str(templates_dir))

FIELD_LABELS: Dict[str, str] = {
    "prompt": "промт для генерации",
    "text": "текст для анализа",
    "question": "уточняющий вопрос",
    "topic": "тема проповеди",
    "bible_text": "библейский фрагмент",
    "occasion": "повод/праздник",
    "audience": "аудитория",
    "style": "стиль",
    "max_new_tokens": "максимальная длина ответа",
    "temperature": "параметр temperature",
    "top_p": "параметр top-p",
    "repetition_penalty": "штраф за повторения",
    "top_k_sources": "количество источников",
}


def _human_field_name(loc: Tuple[Any, ...]) -> str:
    useful = [str(x) for x in loc if isinstance(x, str) and x not in {"body", "query", "path"}]
    if not useful:
        return "запрос"
    return FIELD_LABELS.get(useful[-1], useful[-1])


def _translate_validation_error(err: Dict[str, Any]) -> str:
    err_type = err.get("type", "")
    field = _human_field_name(tuple(err.get("loc", ())))
    ctx = err.get("ctx") or {}
    raw_msg = str(err.get("msg") or "").strip()

    if err_type == "string_too_short":
        min_len = ctx.get("min_length")
        return f"Поле «{field}» слишком короткое. Минимум: {min_len} символов."
    if err_type == "string_too_long":
        max_len = ctx.get("max_length")
        return f"Поле «{field}» слишком длинное. Максимум: {max_len} символов."
    if err_type == "missing":
        return f"Поле «{field}» обязательно для заполнения."
    if err_type in {"float_parsing", "int_parsing", "bool_parsing"}:
        return f"Поле «{field}» имеет неверный формат."
    if err_type == "greater_than_equal":
        return f"Поле «{field}» должно быть не меньше {ctx.get('ge')}."
    if err_type == "less_than_equal":
        return f"Поле «{field}» должно быть не больше {ctx.get('le')}."
    if err_type.startswith("value_error") and raw_msg:
        return raw_msg

    # Fallback, если тип ошибки не сопоставлен.
    return f"Некорректное значение поля «{field}»."


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details: List[str] = [_translate_validation_error(err) for err in exc.errors()]
    return JSONResponse(
        status_code=422,
        content={
            "error": "Некорректные данные запроса.",
            "details": details,
            "code": "VALIDATION_ERROR",
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Ошибка выполнения запроса."
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": detail,
            "code": "HTTP_ERROR",
        },
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_loaded, adapter_loaded = assistant.health_flags()
    return HealthResponse(
        status="ok",
        model_loaded=model_loaded,
        base_model_name=settings.base_model_name,
        adapter_loaded=adapter_loaded,
    )


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return assistant.analyze(req)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    return assistant.generate_sermon(req)


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "dev",
    )
