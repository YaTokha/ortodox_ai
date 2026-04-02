from __future__ import annotations

import asyncio
from datetime import date
from html import unescape
import re
import time
from typing import Any, Dict, List, Optional

import httpx


class OrthodoxCalendarService:
    _FIXED_FEASTS: Dict[str, str] = {
        "01-07": "Рождество Христово",
        "01-19": "Крещение Господне (Богоявление)",
        "02-15": "Сретение Господне",
        "04-07": "Благовещение Пресвятой Богородицы",
        "08-19": "Преображение Господне",
        "08-28": "Успение Пресвятой Богородицы",
        "09-21": "Рождество Пресвятой Богородицы",
        "09-27": "Воздвижение Креста Господня",
        "12-04": "Введение во храм Пресвятой Богородицы",
    }

    _MONTHS_RU = {
        1: "января",
        2: "февраля",
        3: "марта",
        4: "апреля",
        5: "мая",
        6: "июня",
        7: "июля",
        8: "августа",
        9: "сентября",
        10: "октября",
        11: "ноября",
        12: "декабря",
    }
    _WEEKDAYS_RU = {
        0: "понедельник",
        1: "вторник",
        2: "среда",
        3: "четверг",
        4: "пятница",
        5: "суббота",
        6: "воскресенье",
    }

    def __init__(self, cache_ttl_seconds: int = 6 * 60 * 60, fallback_cache_ttl_seconds: int = 90):
        self.cache_ttl_seconds = max(60, int(cache_ttl_seconds))
        # fallback-календарь кэшируем кратко, чтобы быстро восстановиться после временного сбоя сети.
        self.fallback_cache_ttl_seconds = max(15, int(fallback_cache_ttl_seconds))
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ts: Dict[str, float] = {}
        self._cache_lock = asyncio.Lock()

    def _strip_html(self, value: Any) -> str:
        if value is None:
            return ""
        text = str(value)
        text = re.sub(r"<[^>]+>", " ", text)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _dedupe_keep_order(self, items: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for raw in items:
            item = self._strip_html(raw)
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
        return out

    async def _fetch_azbyka_day_async(self, day: date) -> Optional[Dict[str, Any]]:
        url = f"https://azbyka.ru/days/api/day/{day.isoformat()}.json"
        headers = {
            "User-Agent": "OrthodoxAI/1.0 (+https://github.com/YaTokha/ortodox_ai)",
            "Accept": "application/json",
        }
        # Два коротких повтора на случай разового DNS/сетевого сбоя.
        for _ in range(2):
            try:
                async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=headers) as client:
                    response = await client.get(url)
                if response.status_code != 200:
                    await asyncio.sleep(0.35)
                    continue
                data = response.json()
                if not isinstance(data, dict):
                    await asyncio.sleep(0.35)
                    continue
                return data
            except Exception:
                await asyncio.sleep(0.35)
                continue
        return None

    def _is_fallback_payload(self, payload: Optional[Dict[str, Any]]) -> bool:
        if not payload:
            return True
        return str(payload.get("source") or "").strip().lower() == "local-fallback"

    def _is_cache_fresh(self, payload: Optional[Dict[str, Any]], ts: float, now: float) -> bool:
        if payload is None:
            return False
        ttl = self.fallback_cache_ttl_seconds if self._is_fallback_payload(payload) else self.cache_ttl_seconds
        return (now - ts) <= ttl

    async def get_day_info_async(self, day: date, force_refresh: bool = False) -> Dict[str, Any]:
        key = day.isoformat()
        now = time.monotonic()

        cached_payload: Optional[Dict[str, Any]] = None
        cached_ts = 0.0
        async with self._cache_lock:
            cached_payload = self._cache.get(key)
            cached_ts = self._cache_ts.get(key, 0.0)
            if not force_refresh and self._is_cache_fresh(cached_payload, cached_ts, now):
                return dict(cached_payload)

        data = await self._fetch_azbyka_day_async(day)
        payload = self._build_day_info(day, data)

        # Если интернет временно недоступен, но есть прошлый "живой" кэш за этот день,
        # возвращаем его вместо локального fallback.
        if self._is_fallback_payload(payload) and cached_payload and not self._is_fallback_payload(cached_payload):
            return dict(cached_payload)

        async with self._cache_lock:
            self._cache[key] = dict(payload)
            self._cache_ts[key] = time.monotonic()
        return payload

    def get_day_info(self, day: date, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Синхронный wrapper для CLI/скриптов и тестов.
        В веб-API используем get_day_info_async().
        """
        try:
            asyncio.get_running_loop()
            # Внутри уже работающего loop не запускаем asyncio.run.
            return self._build_day_info(day, None)
        except RuntimeError:
            return asyncio.run(self.get_day_info_async(day, force_refresh=force_refresh))

    def _extract_feasts(self, payload: Dict[str, Any]) -> List[str]:
        holidays = payload.get("holidays")
        if not isinstance(holidays, list):
            return []
        titles = []
        for item in holidays:
            if not isinstance(item, dict):
                continue
            title = self._strip_html(item.get("title"))
            if title:
                titles.append(title)
        return self._dedupe_keep_order(titles)

    def _extract_saints(self, payload: Dict[str, Any], limit: int = 12) -> List[str]:
        saints = payload.get("saints")
        if not isinstance(saints, list):
            return []
        out: List[str] = []
        for saint in saints:
            if not isinstance(saint, dict):
                continue
            title = self._strip_html(saint.get("title"))
            if not title:
                continue
            prefix = self._strip_html(saint.get("type_of_sanctity") or saint.get("prefix"))
            if prefix and not title.lower().startswith(prefix.lower()):
                out.append(f"{prefix} {title}")
            else:
                out.append(title)
        result = self._dedupe_keep_order(out)
        return result[:limit]

    def _extract_fasting(self, payload: Dict[str, Any]) -> str:
        fasting = payload.get("fasting")
        if not isinstance(fasting, dict):
            return ""

        parts = []
        for key in ["round_week", "weeks", "fasting", "description"]:
            value = self._strip_html(fasting.get(key))
            if value:
                parts.append(value)
        unique = self._dedupe_keep_order(parts)
        return ". ".join(unique)

    def _format_date_ru(self, day: date) -> str:
        weekday = self._WEEKDAYS_RU.get(day.weekday(), "")
        month = self._MONTHS_RU.get(day.month, "")
        return f"{day.day:02d} {month} {day.year}, {weekday}".strip(", ")

    def _build_day_info(self, day: date, data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        source = "azbyka.ru" if data else "local-fallback"

        feasts = self._extract_feasts(data or {})
        saints = self._extract_saints(data or {})
        fasting = self._extract_fasting(data or {})

        mmdd = f"{day.month:02d}-{day.day:02d}"
        fixed_feast = self._FIXED_FEASTS.get(mmdd)
        if fixed_feast and fixed_feast not in feasts:
            feasts.insert(0, fixed_feast)

        if day.weekday() == 6 and "Воскресный день" not in feasts:
            feasts.append("Воскресный день")

        if not saints:
            saints = ["Память святых дня"]

        main_feast = feasts[0] if feasts else None
        topic_of_day = main_feast or saints[0] or "Память святых дня"

        return {
            "date_iso": day.isoformat(),
            "date_ru": self._format_date_ru(day),
            "topic_of_day": topic_of_day,
            "main_feast": main_feast,
            "feasts": feasts[:4],
            "saints": saints[:12],
            "fasting": fasting or None,
            "source": source,
        }
