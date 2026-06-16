from datetime import date
from pathlib import Path
from collections import defaultdict, deque
import asyncio
import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    CalendarDayResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    HumanHealthResponse,
)
from app.services.assistant_service import OrthodoxAssistantService
from app.services.calendar_service import OrthodoxCalendarService

settings = get_settings()
assistant = OrthodoxAssistantService(settings)
calendar_service = OrthodoxCalendarService()

logger = logging.getLogger("orthodox_ai.api")


class _RequestIdFallbackFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True


if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s")
for handler in logging.getLogger().handlers:
    handler.addFilter(_RequestIdFallbackFilter())

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

RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_GENERATE_PER_WINDOW = 24
_rate_limit_buckets: Dict[str, deque] = defaultdict(deque)
START_MONOTONIC = time.monotonic()

GENERATE_CACHE_TTL_SECONDS = 180
GENERATE_CACHE_MAX_ITEMS = 100
_generate_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_generate_inflight: Dict[str, asyncio.Future] = {}
_generate_lock = asyncio.Lock()

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


def _get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "")


def _client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    if forwarded_for:
        return forwarded_for
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _check_generate_rate_limit(client_key: str) -> bool:
    now = time.monotonic()
    bucket = _rate_limit_buckets[client_key]
    while bucket and (now - bucket[0]) > RATE_LIMIT_WINDOW_SECONDS:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_GENERATE_PER_WINDOW:
        return False
    bucket.append(now)
    return True


def _uptime_human(seconds: int) -> str:
    sec = max(0, int(seconds))
    if sec < 60:
        return f"{sec} сек."
    minutes, s = divmod(sec, 60)
    if minutes < 60:
        return f"{minutes} мин. {s} сек."
    hours, m = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} ч. {m} мин."
    days, h = divmod(hours, 24)
    return f"{days} дн. {h} ч."


def _make_generate_cache_key(req: GenerateRequest) -> str:
    payload = req.model_dump(mode="json")
    packed = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def _prune_generate_cache(now: float) -> None:
    expired = [k for k, (ts, _) in _generate_cache.items() if (now - ts) > GENERATE_CACHE_TTL_SECONDS]
    for key in expired:
        _generate_cache.pop(key, None)
    if len(_generate_cache) <= GENERATE_CACHE_MAX_ITEMS:
        return
    # Удаляем самые старые элементы, если кэш переполнен.
    oldest = sorted(_generate_cache.items(), key=lambda item: item[1][0])
    overflow = len(_generate_cache) - GENERATE_CACHE_MAX_ITEMS
    for key, _ in oldest[:overflow]:
        _generate_cache.pop(key, None)


async def _get_cached_generate(cache_key: str) -> Optional[Dict[str, Any]]:
    now = time.monotonic()
    async with _generate_lock:
        _prune_generate_cache(now)
        item = _generate_cache.get(cache_key)
        if not item:
            return None
        ts, payload = item
        if (now - ts) > GENERATE_CACHE_TTL_SECONDS:
            _generate_cache.pop(cache_key, None)
            return None
        return dict(payload)


async def _set_cached_generate(cache_key: str, payload: Dict[str, Any]) -> None:
    now = time.monotonic()
    async with _generate_lock:
        _prune_generate_cache(now)
        _generate_cache[cache_key] = (now, dict(payload))


async def _acquire_generate_inflight(cache_key: str) -> Tuple[asyncio.Future, bool]:
    async with _generate_lock:
        existing = _generate_inflight.get(cache_key)
        if existing is not None:
            return existing, False
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        _generate_inflight[cache_key] = fut
        return fut, True


async def _release_generate_inflight(cache_key: str, fut: asyncio.Future) -> None:
    async with _generate_lock:
        current = _generate_inflight.get(cache_key)
        if current is fut:
            _generate_inflight.pop(cache_key, None)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    incoming = request.headers.get("x-request-id", "").strip()
    request_id = incoming or str(uuid4())
    request.state.request_id = request_id
    request.state.request_started_at = time.monotonic()
    response = await call_next(request)
    process_ms = (time.monotonic() - request.state.request_started_at) * 1000
    response.headers["x-request-id"] = request_id
    response.headers["x-process-time-ms"] = f"{process_ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details: List[str] = [_translate_validation_error(err) for err in exc.errors()]
    request_id = _get_request_id(request)
    logger.warning(
        "validation_error path=%s details=%s",
        request.url.path,
        details,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=422,
        content={
            "error": "Некорректные данные запроса.",
            "details": details,
            "code": "VALIDATION_ERROR",
            "request_id": request_id,
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else "Ошибка выполнения запроса."
    request_id = _get_request_id(request)
    logger.warning(
        "http_error path=%s status=%s detail=%s",
        request.url.path,
        exc.status_code,
        detail,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": detail,
            "code": "HTTP_ERROR",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = _get_request_id(request)
    logger.exception(
        "unhandled_error path=%s type=%s",
        request.url.path,
        exc.__class__.__name__,
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "Внутренняя ошибка сервиса. Повторите запрос позже.",
            "code": "INTERNAL_ERROR",
            "request_id": request_id,
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


@app.get("/api/health/human", response_model=HumanHealthResponse)
async def health_human() -> HumanHealthResponse:
    model_loaded, adapter_loaded = assistant.health_flags()
    uptime_seconds = int(time.monotonic() - START_MONOTONIC)
    adapter_path = (settings.lora_adapter_path or "").strip()
    adapter_configured = bool(adapter_path)
    adapter_exists = Path(adapter_path).exists() if adapter_configured else False

    if settings.disable_model:
        generation_status = "Ограниченный режим: загрузка модели отключена параметром DISABLE_MODEL=true."
    elif model_loaded and adapter_loaded:
        generation_status = "Полностью готов к генерации (модель и адаптер подключены)."
    elif model_loaded and not adapter_loaded:
        generation_status = "Готов к генерации на базовой модели (адаптер не подключен)."
    elif adapter_configured and adapter_exists:
        generation_status = (
            "Модель еще не загружена в память. Это штатно: при первом запросе генерации "
            "она будет загружена, и подключится адаптер дообучения."
        )
    else:
        generation_status = (
            "Модель еще не загружена в память. При первом запросе генерации она будет "
            "подгружена автоматически."
        )

    return HumanHealthResponse(
        service_status="Сервис работает",
        generation_status=generation_status,
        model_name=settings.base_model_name,
        model_loaded=model_loaded,
        adapter_loaded=adapter_loaded,
        rate_limit_note=(
            f"Лимит генерации: до {RATE_LIMIT_GENERATE_PER_WINDOW} запросов за "
            f"{RATE_LIMIT_WINDOW_SECONDS} секунд с одного IP."
        ),
        uptime_seconds=uptime_seconds,
        uptime_human=_uptime_human(uptime_seconds),
    )


@app.get("/api/calendar/day", response_model=CalendarDayResponse)
async def calendar_day(
    day: Optional[str] = Query(default=None, description="Дата в формате YYYY-MM-DD"),
    force_refresh: bool = Query(default=False, description="Игнорировать кэш и обновить данные из источника"),
) -> CalendarDayResponse:
    target_day = date.today()
    if day:
        try:
            target_day = date.fromisoformat(day)
        except ValueError:
            raise HTTPException(status_code=422, detail="Некорректная дата. Используйте формат YYYY-MM-DD.")

    payload = await calendar_service.get_day_info_async(target_day, force_refresh=force_refresh)
    return CalendarDayResponse(**payload)


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    return assistant.analyze(req)


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: Request, req: GenerateRequest) -> GenerateResponse:
    request_id = _get_request_id(request)
    client_key = _client_key(request)
    if not _check_generate_rate_limit(client_key):
        raise HTTPException(
            status_code=429,
            detail="Слишком много запросов к генерации. Подождите немного и повторите попытку.",
        )
    logger.info(
        "generate_start client=%s has_prompt=%s has_topic=%s",
        client_key,
        bool((req.prompt or "").strip()),
        bool((req.topic or "").strip()),
        extra={"request_id": request_id},
    )
    cache_key = _make_generate_cache_key(req)
    use_cache = bool(settings.enable_generate_cache)
    if use_cache:
        cached = await _get_cached_generate(cache_key)
        if cached is not None:
            logger.info("generate_cache_hit client=%s", client_key, extra={"request_id": request_id})
            return GenerateResponse(**cached)

    inflight, is_owner = await _acquire_generate_inflight(cache_key)
    if not is_owner:
        logger.info("generate_wait_inflight client=%s", client_key, extra={"request_id": request_id})
        payload = await inflight
        if isinstance(payload, dict) and payload.get("__error__"):
            raise HTTPException(
                status_code=500,
                detail="Не удалось выполнить генерацию проповеди. Повторите попытку.",
            )
        return GenerateResponse(**payload)

    try:
        generated = await asyncio.to_thread(assistant.generate_sermon, req)
        payload = generated.model_dump(mode="json")
        if use_cache:
            await _set_cached_generate(cache_key, payload)
        inflight.set_result(payload)
        return generated
    except Exception as exc:
        if not inflight.done():
            inflight.set_result({"__error__": "failed"})
        raise
    finally:
        await _release_generate_inflight(cache_key, inflight)


def run() -> None:
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "dev",
    )
