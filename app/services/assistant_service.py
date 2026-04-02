import re
import hashlib
import random
from html import unescape
from typing import Dict, List, Optional, Tuple

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, Citation, GenerateRequest, GenerateResponse, QualityMetrics
from app.services.generation import SermonGenerator
from app.services.retrieval import CorpusRetrievalService
from app.services.text_preprocessor import TextPreprocessor

DISCLAIMER = (
    "Материал сгенерирован ИИ и предназначен как черновик для подготовки. "
    "Перед использованием требуется богословская проверка священнослужителем."
)

CLICHE_MARKERS = (
    "пусть в нашем дне будет место для тишины перед богом",
    "если же мы падаем, не будем отчаиваться",
    "путь спасения совершается не в безошибочности",
    "проверим себя:",
    "с благодарением богу продолжим путь христианской жизни",
)


class OrthodoxAssistantService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = TextPreprocessor()
        self.retrieval = CorpusRetrievalService(settings.corpus_abspath())
        self.generator = SermonGenerator(settings)
        self._recent_sermon_signatures: List[str] = []
        self._recent_sermons: List[str] = []
        self._recent_choice_index: Dict[str, int] = {}
        self._rotation_state: Dict[str, Dict[str, object]] = {}

    def _build_analysis_prompt(self, req: AnalyzeRequest, citations: List[Citation]) -> str:
        sources_block = "\n".join(
            f"- {c.source_type}; {c.author or 'не указан'}; {c.reference or c.title or c.id}: {c.excerpt}"
            for c in citations
        )

        return (
            "Ты православный богословский ассистент. Выполни аккуратный анализ фрагмента без догматических новшеств.\n"
            "Структура ответа: 1) историко-культурный контекст, 2) святоотеческое толкование, "
            "3) практическое применение для христианской жизни.\n"
            f"Вопрос пользователя: {req.question or 'Общий анализ фрагмента'}\n"
            f"Фрагмент: {req.text}\n"
            f"Опорные источники:\n{sources_block}\n"
            "Ответ:"
        )

    def _build_sermon_prompt(self, req: GenerateRequest, citations: List[Citation]) -> str:
        strict_sin_line = ""
        if self._is_sin_topic(req):
            strict_sin_line = (
                "3a) Для темы о грехах и страстях используй строгий, трезвенный и обличительный тон без грубости; "
                "обязательно назови грех, его духовные последствия и путь исправления через покаяние.\n"
            )
        marriage_line = ""
        marriage_low = self.preprocessor.normalize(f"{req.topic or ''} {req.prompt or ''}").lower()
        if self._is_marriage_topic_low(marriage_low):
            marriage_line = (
                "3b) Для темы о Таинстве Венчания обязательно раскрой смысл церковного брака, "
                "взаимную ответственность супругов, верность, жертвенную любовь и совместную молитву в семье.\n"
            )
        return (
            "Напиши цельную православную проповедь на русском языке.\n"
            "Требования:\n"
            "1) Верни только готовый связный текст проповеди.\n"
            "2) Структура: вступление, основная часть, заключение (цельными абзацами).\n"
            "3) Тон пастырский, спокойный, назидательный.\n"
            f"{strict_sin_line}"
            f"{marriage_line}"
            "4) Начало: «Во имя Отца, и Сына, и Святого Духа!» и обращение к пастве.\n"
            "5) Основа: евангельский смысл и святоотеческая традиция в пересказе, без прямых цитат.\n"
            "6) Завершение: практический призыв к покаянию/добрым делам и финал «Аминь.»\n"
            "7) Не вставляй ссылки, служебные метки, названия сайтов и технические пометки.\n"
            "8) Не делай списки и перечисления через «;», пиши плавными связными фразами.\n"
            "9) Каждый абзац логически развивает предыдущий: добавляй причинно-следственные переходы и не допускай разрозненных мыслей.\n"
            f"Тема: {req.topic}\n"
            f"Повод/праздник: {req.occasion or 'не указан'}\n"
            f"Аудитория: {req.audience or 'приход'}\n"
            f"Стиль: {req.style}\n"
            f"Библейский фрагмент: {req.bible_text or 'не указан'}\n"
            "Опора: Священное Писание и святоотеческая православная традиция.\n"
            "Проповедь:"
        )

    def _build_user_prompt_mode(self, req: GenerateRequest, citations: List[Citation]) -> str:
        user_prompt = self.preprocessor.normalize(req.prompt or "")
        strict_sin_line = ""
        if self._is_sin_topic_low(user_prompt.lower()):
            strict_sin_line = (
                "Для темы о грехах и страстях держи строгий, трезвенный и обличительный тон без грубости; "
                "покажи тяжесть греха и необходимость покаяния.\n"
            )
        marriage_line = ""
        if self._is_marriage_topic_low(user_prompt.lower()):
            marriage_line = (
                "Для темы о Таинстве Венчания пиши конкретно о христианском браке: "
                "верность супругов, жертвенная любовь, совместная молитва и ответственность перед Богом.\n"
            )
        return (
            "Ты православный ассистент для подготовки проповедей.\n"
            "Сгенерируй цельную проповедь по запросу пользователя.\n"
            "Структура: вступление, основная часть, заключение (абзацы, не списки).\n"
            "Начало: «Во имя Отца, и Сына, и Святого Духа!» и обращение к пастве.\n"
            f"{strict_sin_line}"
            f"{marriage_line}"
            f"Предпочтительный стиль: {req.style or 'пастырский'}.\n"
            "Без прямых цитат; передавай смысл Писания и святых отцов своими словами.\n"
            "Не используй перечисления через «;», служебные пометки и телеграфный стиль.\n"
            "Не добавляй ссылки, имена файлов, названия сайтов, метки типа commentary/sermon и технические вставки.\n"
            "Сделай текст внутренне цельным: пусть абзацы связно переходят друг в друга и углубляют тему, а не повторяют одно и то же.\n"
            "Опирайся на Священное Писание и святоотеческую православную традицию.\n"
            f"Запрос пользователя: {user_prompt}\n"
            "Проповедь:"
        )

    def _build_outline(self, text: str) -> List[str]:
        sentences = self.preprocessor.split_into_sentences(text)
        if not sentences:
            return ["Вступление", "Толкование", "Практические выводы", "Заключение"]
        return [s[:140] for s in sentences[:4]]

    def _cleanup_sermon_text(self, text: str) -> str:
        text = unescape(text or "")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"#_\d+(?:-\d+)?(?:_\d+)?", " ", text)
        text = text.replace("\u00a0", " ")
        if not text:
            return text

        # Если модель всё же вернула служебный пролог, отрезаем его.
        for marker in ["Проповедь:", "Ответ:"]:
            if marker in text:
                text = text.split(marker, 1)[1].strip()

        # Если модель вернула формат "План/Текст/Источники", оставляем только блок после "Текст:".
        text_marker = re.search(r"(?:^|\n)\s*Текст:\s*", text)
        if text_marker:
            text = text[text_marker.end() :].strip()
        for tail in ["\nИсточники:", "\nМодель:", "\nПримечание:", "\nПлан:"]:
            idx = text.find(tail)
            if idx != -1:
                text = text[:idx].strip()

        # Убираем служебные заголовки и лишние хвосты.
        bad_prefixes = ("План:", "Текст:", "Источники:", "Модель:", "Примечание:")
        cleaned_lines = []
        prev_clean = ""
        for line in text.split("\n"):
            ln = re.sub(r"\s+", " ", line).strip()
            if not ln:
                cleaned_lines.append("")
                continue
            low_ln = ln.lower()
            if low_ln == prev_clean:
                continue
            if re.match(r"^правило\s*\d+\s*[:.]", low_ln):
                continue
            if re.match(r"^\d{6,}\s+", low_ln):
                continue
            if any(
                marker in low_ln
                for marker in [
                    "азбука веры",
                    "сретенский монастырь",
                    "свято-елисаветинский женский монастырь",
                    "pravoslavie.ru",
                    "православие.ру",
                    "royallib",
                    "livejournal",
                    "livej",
                    "отдыхая с пользой",
                ]
            ):
                continue
            if re.match(r"^(-|\*|•)?\s*(commentary|sermon|bible|analysis)\s*;", low_ln):
                continue
            if ("источник:" in low_ln or "source:" in low_ln) and low_ln.count(";") >= 2:
                continue
            if any(
                marker in low_ln
                for marker in [
                    "style definitions",
                    "mso-",
                    "p.msonormal",
                    "div.msonormal",
                    "@page section",
                    "section1",
                    "font-family",
                    "times new roman",
                ]
            ):
                continue
            if ln.startswith(bad_prefixes):
                continue
            if re.match(r"^(-|\*|•)\s+", ln):
                continue
            if re.match(r"^\d+[\.\)]\s+", ln):
                continue
            cleaned_lines.append(line)
            prev_clean = low_ln
        text = "\n".join(cleaned_lines).strip()

        # Убираем искусственный обрыв вроде "С" или одиночных маркеров.
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _is_noisy_sermon(self, text: str, require_structure_markers: bool = True) -> bool:
        if not text:
            return True
        low = text.lower()
        structural_markers = [
            "план:",
            "текст:",
            "источники:",
            "модель:",
            "примечание:",
            "черновик проповеди",
            "fallback-режим",
        ]
        if any(marker in low for marker in structural_markers):
            return True

        noisy_markers = [
            "http://",
            "https://",
            "facebook.com",
            "vk.com",
            "instagram.com",
            "youtube.com",
            "<!--",
            "&lt;!--",
            "mso-",
            "style definitions",
            "@page section",
            "p.msonormal",
            "commentary;",
            "sermon;",
            "#_",
            "source:",
            "источник:",
            "royallib",
            "livejournal",
            "livej",
            "правило 1:",
            "правило 2:",
        ]
        if any(m in low for m in noisy_markers):
            return True

        meta_markers = [
            "используй для своих целей",
            "не забудь",
            "по тексту",
            "как правило",
            "учитесь читать",
            "описывай ",
            "не смешивай",
            "отталкивайся от",
            "говорите обо всем подробно",
            "не повторяй текст",
            "записывай",
            "предоставь",
            "передай",
            "пользовател",
            "обсудить с тобой",
            "инструкция",
            "задание",
            "по возможности",
            "не пытайся",
            "не размещай",
            "в начале нужно",
            "проповеди должны",
            "не забывай",
            "рассказывай",
            "подчеркивай",
            "обращать внимание",
            "какие слова",
            "какие выражения",
            "на что обращать",
            "как ее принять",
            "как её принять",
        ]
        if any(m in low for m in meta_markers):
            return True

        total = max(len(text), 1)
        digits_ratio = sum(ch.isdigit() for ch in text) / total
        latin_ratio = sum(("a" <= ch.lower() <= "z") for ch in text) / total
        if digits_ratio > 0.06 or latin_ratio > 0.15:
            return True

        nonempty_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(nonempty_lines) >= 4:
            list_like = sum(bool(re.match(r"^(-|\*|•|\d+[\.\)])\s+", ln)) for ln in nonempty_lines)
            if list_like / len(nonempty_lines) > 0.35:
                return True

        if len(nonempty_lines) >= 6:
            short_fragments = sum(
                1
                for ln in nonempty_lines
                if len(ln.split()) <= 6 and not any(p in ln for p in ".!?")
            )
            if short_fragments / len(nonempty_lines) > 0.5:
                return True

        if sum(ch in ".!?" for ch in text) < 3:
            return True

        if re.search(r"(^|\n)\s*(-|\*|•)?\s*(commentary|sermon|bible|analysis)\s*;", low):
            return True

        if re.search(r"(^|\n)\s*правило\s*\d+\s*[:.]", low):
            return True

        if require_structure_markers:
            if not re.search(r"\b(возлюбленн|братья и сестры)\b", low):
                return True
            if "амин" not in low[-260:]:
                return True
            intro, _, _ = self._split_sermon_sections(text)
            intro_low = (intro or "").lower()
            if not intro_low.startswith("во имя отца"):
                return True
            sal_idx = intro_low.find("дорогие братья и сестры")
            if sal_idx == -1 or sal_idx > 140:
                return True

        # Для итоговой проповеди требуем более длинный текст, для черновика мягче.
        min_len = 220 if require_structure_markers else 120
        if len(text) < min_len:
            return True
        return False

    def _is_extreme_noise(self, text: str) -> bool:
        low = (text or "").lower()
        if not low.strip():
            return True
        hard_markers = [
            "http://",
            "https://",
            "commentary;",
            "source:",
            "источник:",
            "royallib",
            "livejournal",
            "livej",
            "<!--",
            "style definitions",
            "mso-",
            "fallback-режим",
        ]
        if any(m in low for m in hard_markers):
            return True
        if re.search(r"(^|\n)\s*правило\s*\d+\s*[:.]", low):
            return True
        if re.search(r"(^|\n)\s*\d{6,}\s+", low):
            return True
        if len(self.preprocessor.split_into_sentences(text)) < 3:
            return True
        return False

    def _has_direct_quotes(self, text: str) -> bool:
        if not text:
            return False
        quote_spans = re.findall(r"[«\"]([^\"»]{20,})[»\"]", text)
        # Одну короткую цитату можем пережить (позже кавычки будут сняты), но массив цитат считаем шумом.
        if len(quote_spans) >= 2:
            return True
        low = text.lower()
        quote_markers = [
            "как сказано",
            "как говорит",
            "сказано:",
            "по слову",
            "цитата",
        ]
        return any(m in low for m in quote_markers) and len(quote_spans) >= 1

    def _compose_title(self, req: GenerateRequest) -> str:
        topic_raw = self._extract_topic(req).strip(" .,:;!?")
        sin_profile = self._sin_profile(topic_raw.lower())
        topic = str(sin_profile.get("title_topic", topic_raw)) if sin_profile else topic_raw
        topic = self._normalize_topic_for_title(topic)
        if not topic:
            topic = "христианской жизни"
        topic = topic[0].upper() + topic[1:] if topic else topic
        topic = self._apply_orthodox_casing(topic)
        return f"Проповедь на тему: «{topic}»"

    def _normalize_topic_for_title(self, topic: str) -> str:
        out = self.preprocessor.normalize(topic or "")
        if not out:
            return out
        fixes = [
            (r"\bгрехе\b", "грех"),
            (r"\bстрасти\b", "страсть"),
            (r"\bпороке\b", "порок"),
            (r"\bо\s+грехе\s+", "грех "),
            (r"\bо\s+страсти\s+", "страсть "),
            (r"\bтаинстве\s+венчан\w*\b", "таинство Венчания"),
            (r"\bо\s+таинстве\s+венчан\w*\b", "таинство Венчания"),
        ]
        for pattern, repl in fixes:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        return out.strip(" .,:;!?")

    def _split_sermon_sections(self, text: str) -> Tuple[str, str, str]:
        low = (text or "").lower()
        m_intro = re.search(r"вступление\.\s*", low)
        m_main = re.search(r"основная часть\.\s*", low)
        m_concl = re.search(r"заключение\.\s*", low)
        if not (m_intro and m_main and m_concl):
            return "", "", ""
        intro = text[m_intro.end() : m_main.start()].strip()
        main = text[m_main.end() : m_concl.start()].strip()
        concl = text[m_concl.end() :].strip()
        return intro, main, concl

    def _is_section_poor(self, section: str, min_words: int) -> bool:
        norm = self.preprocessor.normalize(section or "")
        words = re.findall(r"[А-Яа-яA-Za-zЁё]+", norm)
        if len(words) < min_words:
            return True
        bad = ["аминь.", "аминь", "основная часть.", "заключение.", "вступление."]
        low = norm.lower()
        if low in bad:
            return True
        if low.count("аминь") >= 1 and len(words) < max(min_words + 8, 24):
            return True
        if norm.count(";") >= 5:
            return True
        if len(re.findall(r"\bтема\b", low)) >= 3:
            return True
        if len(re.findall(r"\bразговор\b", low)) >= 3:
            return True
        if norm.count(":") >= 6:
            return True
        # Слишком плотные повторы слов -> низкое качество.
        if words:
            uniq_ratio = len({w.lower() for w in words}) / len(words)
            if uniq_ratio < 0.33:
                return True
        # Перегруженные предложения на 80+ слов почти всегда выглядят как "поток".
        sents = self.preprocessor.split_into_sentences(norm)
        if sents:
            too_long = sum(1 for s in sents if len(s.split()) >= 80)
            if too_long >= 1:
                return True
        return False

    def _format_three_part_sermon(self, text: str, req: GenerateRequest, citations: List[Citation]) -> str:
        plain = self.preprocessor.normalize(text or "")
        # Убираем возможные заголовки-разделы, если они уже пришли от модели.
        plain = re.sub(r"\bВступление\s*[:.]\s*", "", plain, flags=re.IGNORECASE)
        plain = re.sub(r"\bОсновная часть\s*[:.]\s*", "", plain, flags=re.IGNORECASE)
        plain = re.sub(r"\bЗаключение\s*[:.]\s*", "", plain, flags=re.IGNORECASE)
        plain = re.sub(r"\bПроповедь\s*[:.]\s*", "", plain, flags=re.IGNORECASE)
        plain = plain.strip()

        # Снимаем кавычки и прямую речь-цитирование.
        plain = plain.replace("«", "").replace("»", "").replace('"', "")
        plain = re.sub(r"\b(как сказано|как говорит|сказано)\s*:\s*", "", plain, flags=re.IGNORECASE)

        if self._is_extreme_noise(plain):
            return self._compose_safe_sermon(req, citations)

        sentences = self.preprocessor.split_into_sentences(plain)
        if len(sentences) < 3:
            return self._compose_safe_sermon(req, citations)

        intro = " ".join(sentences[: min(2, len(sentences))]).strip()
        if len(sentences) >= 5:
            main = " ".join(sentences[2:-2]).strip() or " ".join(sentences[2:4]).strip()
            concl = " ".join(sentences[-2:]).strip()
        elif len(sentences) == 4:
            main = sentences[2].strip()
            concl = sentences[3].strip()
        else:
            main = sentences[1].strip()
            concl = sentences[2].strip()

        if "во имя отца" not in intro.lower():
            intro = "Во имя Отца, и Сына, и Святого Духа! " + intro
        if "дорогие братья и сестры" not in intro.lower() and "возлюбленные братья и сестры" not in intro.lower():
            intro = intro + " Дорогие братья и сестры!"

        if "амин" not in concl.lower():
            concl = concl.rstrip(". ") + ". Аминь."

        title = self._compose_title(req)
        sermon = (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{concl}"
        )
        return self._ensure_amen_last(sermon, req)

    def _is_structured_sermon(self, text: str) -> bool:
        low = (text or "").lower()
        if not (low.startswith("проповедь:") or low.startswith("проповедь на тему:")):
            return False
        required = [
            "вступление.",
            "основная часть.",
            "заключение.",
            "во имя отца, и сына, и святого духа!",
            "аминь",
        ]
        if not all(x in low for x in required):
            return False
        if re.search(r"(^|\n)\s*правило\s*\d+\s*[:.]", low):
            return False
        if any(x in low for x in ["royallib", "livej", "livejournal", "commentary;", "источник:"]):
            return False
        intro, main, concl = self._split_sermon_sections(text)
        if not (intro and main and concl):
            return False
        if self._is_section_poor(intro, min_words=16):
            return False
        if self._is_section_poor(main, min_words=45):
            return False
        if self._is_section_poor(concl, min_words=12):
            return False
        if sum(ch in ".!?" for ch in main) < 3:
            return False
        if text.count(";") >= 12:
            return False
        if text.count(":") >= 16:
            return False
        if re.search(r"#_\d+(?:-\d+)?(?:_\d+)?", text):
            return False
        return True

    def _extract_topic(self, req: GenerateRequest) -> str:
        topic = self.preprocessor.normalize(req.topic or "")
        if topic:
            return topic

        prompt = self.preprocessor.normalize(req.prompt or "")
        if not prompt:
            return "христианская жизнь"

        # Убираем служебные хвосты, если они по ошибке попали в prompt.
        prompt = re.sub(r"\bсделай\s+проповед\w*[^.!?\n]*", "", prompt, flags=re.IGNORECASE)
        prompt = re.sub(r"\bвариант\s*[abаб]\s*:[^.!?\n]*", "", prompt, flags=re.IGNORECASE)
        prompt = prompt.strip()

        # Убираем типичные императивные префиксы пользовательского промта.
        patterns = [
            r"^(сгенерируй|составь|подготовь|напиши|создай)\s+",
            r"^(кратк\w+|цельн\w+)\s+",
            r"^(православн\w+)\s+",
            r"^проповед\w*\s+(о|про|на тему)\s+",
            r"^проповед\w*\s+опираясь\s+на\s+",
            r"^опираясь\s+на\s+",
            r"^притч\w+\s+о\s+",
        ]
        topic_guess = prompt
        for p in patterns:
            topic_guess = re.sub(p, "", topic_guess, flags=re.IGNORECASE)
        # Тема обычно содержится в первой фразе; хвосты инструкций отсекаем.
        topic_guess = re.split(r"[.!?\n]", topic_guess, maxsplit=1)[0]
        topic_guess = re.sub(r"\bсделай\b.*$", "", topic_guess, flags=re.IGNORECASE)
        topic_guess = re.sub(r"\bвариант\s*[abаб]\b.*$", "", topic_guess, flags=re.IGNORECASE)
        topic_guess = topic_guess.strip(" .,:;!-?")

        if len(topic_guess) < 3:
            return "христианская жизнь"
        return topic_guess

    def _apply_orthodox_casing(self, text: str) -> str:
        out = text or ""
        rules = [
            (r"\bвоскресение христово\b", "Воскресение Христово"),
            (r"\bвоскресении христовом\b", "Воскресении Христовом"),
            (r"\bхристов([а-яё]{0,4})\b", r"Христов\1"),
            (r"\bхристос воскресе\b", "Христос Воскресе"),
            (r"\bвоистину воскресе\b", "Воистину Воскресе"),
            (r"\bхристос\b", "Христос"),
            (r"\bпасха\b", "Пасха"),
            (r"\bпресвятая богородица\b", "Пресвятая Богородица"),
            (r"\bпресвятой богородице\b", "Пресвятой Богородице"),
            (r"\bбожия матерь\b", "Божия Матерь"),
            (r"\bбожией матери\b", "Божией Матери"),
            (r"\bбогородиц([а-яё]*)\b", r"Богородиц\1"),
        ]
        for pattern, repl in rules:
            out = re.sub(pattern, repl, out, flags=re.IGNORECASE)
        return out

    def _is_paschal_topic_low(self, topic_low: str) -> bool:
        low = (topic_low or "").lower()
        if "лазар" in low:
            return False
        if "пасх" in low or "христос воскрес" in low:
            return True
        if re.search(r"\bвоскресени[ея]\s+христ", low):
            return True
        if "воскрс" in low and "христ" in low:
            return True
        if "воскрес" in low and "христ" in low:
            return True
        return False

    def _is_resurrection_topic(self, req: GenerateRequest) -> bool:
        topic_low = self._extract_topic(req).lower()
        return self._is_paschal_topic_low(topic_low)

    def _is_lazarus_topic(self, req: GenerateRequest) -> bool:
        topic_low = self._extract_topic(req).lower()
        return any(
            w in topic_low
            for w in [
                "лазар",
                "лазарев",
                "лазарева суббот",
                "воскрешен лазар",
                "четвероднев",
                "вифан",
            ]
        )

    def _is_prodigal_topic(self, req: GenerateRequest) -> bool:
        topic_low = self._extract_topic(req).lower()
        return any(
            w in topic_low
            for w in [
                "блудн",
                "притч о блудн",
                "притча о блудн",
                "притче о блудн",
                "лук 15",
                "сын расточ",
            ]
        )

    def _is_sin_topic_low(self, topic_low: str) -> bool:
        low = (topic_low or "").lower()
        if not low:
            return False
        if re.search(r"\b(семь|7)\s+смертн\w+\s+грех", low):
            return True
        return any(
            w in low
            for w in [
                "грех",
                "грехов",
                "страст",
                "покаян",
                "исповед",
                "осужд",
                "гордын",
                "гнев",
                "завист",
                "уныни",
                "блуд",
                "сребролюб",
                "сквернослов",
                "лжи",
                "лож",
                "пьян",
                "чревоугод",
                "леност",
                "самолюб",
            ]
        )

    def _is_sin_topic(self, req: GenerateRequest) -> bool:
        topic_low = self._extract_topic(req).lower()
        prompt_low = self.preprocessor.normalize(req.prompt or "").lower()
        return self._is_sin_topic_low(topic_low) or self._is_sin_topic_low(prompt_low)

    def _sin_subtopic(self, topic_low: str) -> str:
        low = (topic_low or "").lower()
        if not self._is_sin_topic_low(low):
            return ""
        if any(w in low for w in ["гадан", "таро", "астролог", "гороскоп", "колдов", "маг", "чарод", "оккульт", "экстрасенс", "волшеб"]):
            return "divination"
        if any(w in low for w in ["гордын", "тщеслав", "самомнен", "превознош"]):
            return "pride"
        if any(w in low for w in ["гнев", "раздраж", "злоб", "ярост"]):
            return "anger"
        if any(w in low for w in ["блуд", "прелюбод", "нечистот", "похот", "разврат"]):
            return "lust"
        if any(w in low for w in ["сребролюб", "алчност", "жадност", "корыст", "лихоим"]):
            return "greed"
        if any(w in low for w in ["завист"]):
            return "envy"
        if any(w in low for w in ["пьян", "алкогол", "винопит"]):
            return "drunkenness"
        if any(w in low for w in ["чревоугод", "объяден", "сластолюб"]):
            return "gluttony"
        if any(w in low for w in ["унын", "отчаян", "леност", "праздност"]):
            return "despondency"
        if any(w in low for w in ["осужд", "клевет", "злослов"]):
            return "judgment"
        if any(w in low for w in ["лжи", "лож", "обман"]):
            return "lying"
        return "generic_sin"

    def _sin_profile(self, topic_low: str) -> Optional[Dict[str, object]]:
        code = self._sin_subtopic(topic_low)
        if not code:
            return None

        profiles: Dict[str, Dict[str, object]] = {
            "divination": {
                "title_topic": "Грех гадания",
                "name_genitive": "греха гадания и обращения к оккультным практикам",
                "keywords": ["гадан", "таро", "гороскоп", "оккульт", "чарод", "волшеб", "покаян", "отреч"],
                "focus": "он подменяет доверие Богу поиском тайного знания и вводит душу в духовный обман",
                "practice": "полностью отказаться от гаданий, гороскопов и любых оккультных практик, исповедовать это как грех и восстановить молитвенную жизнь",
                "old_testament": (
                    "Не должен находиться у тебя проводящий сына своего или дочь свою через огонь, прорицатель, гадатель, ворожей, чародей. Ибо мерзок пред Господом всякий, делающий это.",
                    "Втор. 18:10-12",
                ),
                "apostle": (
                    "Дела плоти известны: ... идолослужение, волшебство, вражда, ссоры, зависть, гнев... Поступающие так Царствия Божия не наследуют.",
                    "Гал. 5:19-21",
                ),
                "father": (
                    "Кто ищет знамений вне Бога, тот отступает от живой веры и ранит собственную душу.",
                    "Свт. Игнатий (Брянчанинов)",
                ),
                "preacher": (
                    "Христианин не спрашивает будущего у тьмы, он вверяет свою жизнь Промыслу Божию и живет покаянием.",
                    "Свт. Иоанн Кронштадтский",
                ),
            },
            "pride": {
                "title_topic": "Грех гордыни",
                "name_genitive": "греха гордыни",
                "keywords": ["гордын", "тщеслав", "самолюб", "превознош", "смирен", "покаян"],
                "focus": "она лишает человека смирения, закрывает сердце для благодати и разрушает братскую любовь",
                "practice": "учиться смирению, принимать замечания без ропота и чаще проверять свои мотивы перед Богом",
                "old_testament": (
                    "Погибели предшествует гордость, и падению - надменность.",
                    "Притч. 16:18",
                ),
                "apostle": (
                    "Бог гордым противится, а смиренным дает благодать. Итак покоритесь Богу; противостаньте диаволу, и убежит от вас.",
                    "Иак. 4:6-7",
                ),
                "father": ("Гордость есть начало всех страстей, а смирение - дверь к благодати.", "Свт. Иоанн Златоуст"),
                "preacher": ("Без смирения даже внешнее благочестие становится тонкой формой самообмана.", "Митрополит Антоний Сурожский"),
            },
            "anger": {
                "title_topic": "Грех гнева",
                "name_genitive": "греха гнева",
                "keywords": ["гнев", "раздраж", "злоб", "ярост", "мир", "прощен", "кротост"],
                "focus": "он ослепляет ум, разрушает мир в семье и делает сердце жестким к ближнему",
                "practice": "останавливать раздражение молитвой, хранить язык и первым идти к примирению",
                "old_testament": (
                    "Долготерпеливый лучше храброго, и владеющий собою лучше завоевателя города.",
                    "Притч. 16:32",
                ),
                "apostle": (
                    "Гнев человека не творит правды Божией. Итак, отложив всякую нечистоту и остаток злобы, в кротости примите насаждаемое слово.",
                    "Иак. 1:20-21",
                ),
                "father": ("Ничто так не охлаждает любовь, как привычка оправдывать собственный гнев.", "Свт. Василий Великий"),
                "preacher": ("Кротость - это сила, которая побеждает зло без ненависти к человеку.", "Свт. Лука (Войно-Ясенецкий)"),
            },
            "lust": {
                "title_topic": "Грех блуда",
                "name_genitive": "греха блуда",
                "keywords": ["блуд", "нечистот", "похот", "прелюбод", "целомудр", "чистот"],
                "focus": "он оскверняет сердце, разрушает верность и искажает дар любви",
                "practice": "хранить целомудрие мыслей, избегать соблазняющих привычек и укрепляться в молитве и воздержании",
                "old_testament": (
                    "Завет положил я с глазами моими, чтобы не помышлять мне о девице.",
                    "Иов. 31:1",
                ),
                "apostle": (
                    "Воля Божия есть освящение ваше, чтобы вы воздерживались от блуда; чтобы каждый из вас умел соблюдать свой сосуд в святости и чести.",
                    "1 Фес. 4:3-4",
                ),
                "father": ("Чистота сердца рождается не от страха, а от любви ко Христу и внутреннего трезвения.", "Свт. Феофан Затворник"),
                "preacher": ("Целомудрие сохраняется там, где человек хранит молитву и не играет с соблазном.", "Протоиерей Александр Мень"),
            },
            "greed": {
                "title_topic": "Грех сребролюбия",
                "name_genitive": "греха сребролюбия",
                "keywords": ["сребролюб", "алчност", "жадност", "корыст", "милосерд", "щедрост"],
                "focus": "он превращает богатство в идола и лишает человека сострадания",
                "practice": "учиться щедрости, вести честную жизнь и помнить о милосердии к нуждающимся",
                "old_testament": (
                    "Корыстолюбивый расстроит дом свой; а ненавидящий подарки будет жить.",
                    "Притч. 15:27",
                ),
                "apostle": (
                    "Корень всех зол есть сребролюбие, которому предавшись, некоторые уклонились от веры и сами себя подвергли многим скорбям.",
                    "1 Тим. 6:10",
                ),
                "father": ("Сребролюбие обещает безопасность, но рождает страх, жесткость и внутреннюю пустоту.", "Свт. Иоанн Златоуст"),
                "preacher": ("Там, где сердце служит деньгам, оно перестает слышать боль ближнего.", "Свт. Тихон Задонский"),
            },
            "envy": {
                "title_topic": "Грех зависти",
                "name_genitive": "греха зависти",
                "keywords": ["завист", "ожесточ", "радост", "любов", "благодар"],
                "focus": "она отравляет душу сравнением и делает человека неспособным к благодарности",
                "practice": "отсекать сравнение с другими, благодарить Бога за дар каждого дня и учиться сорадованию",
                "old_testament": ("Кроткое сердце - жизнь для тела, а зависть - гниль для костей.", "Притч. 14:30"),
                "apostle": (
                    "Где зависть и сварливость, там неустройство и все худое. Но мудрость, сходящая свыше, чиста, мирна, скромна, послушлива.",
                    "Иак. 3:16-17",
                ),
                "father": ("Зависть скорбит о благе ближнего и потому первой поражает самого завистника.", "Свт. Василий Великий"),
                "preacher": ("Лекарство от зависти - благодарение Богу и любовь, которая умеет радоваться за другого.", "Свт. Иоанн Кронштадтский"),
            },
            "drunkenness": {
                "title_topic": "Грех пьянства",
                "name_genitive": "греха пьянства",
                "keywords": ["пьян", "алкогол", "винопит", "трезвен", "зависим"],
                "focus": "он порабощает волю, разрушает семью и лишает человека духовной трезвости",
                "practice": "принять подвиг трезвения, просить помощи Церкви и последовательно отказываться от зависимости",
                "old_testament": (
                    "У кого вой? у кого стон? ... у тех, которые долго сидят за вином. Не смотри на вино, как оно краснеет... впоследствии, как змей, оно укусит.",
                    "Притч. 23:29-32",
                ),
                "apostle": (
                    "Не упивайтесь вином, от которого бывает распутство; но исполняйтесь Духом.",
                    "Еф. 5:18",
                ),
                "father": ("Пьянство помрачает ум и делает душу беззащитной перед прочими страстями.", "Свт. Иоанн Златоуст"),
                "preacher": ("Трезвость - это путь к свободе, в которой человек снова учится жить перед Богом и людьми.", "Свт. Лука (Войно-Ясенецкий)"),
            },
            "gluttony": {
                "title_topic": "Грех чревоугодия",
                "name_genitive": "греха чревоугодия",
                "keywords": ["чревоугод", "объяден", "невоздерж", "пост", "трезвен"],
                "focus": "он приучает человека служить удовольствию тела и ослабляет духовную собранность",
                "practice": "хранить воздержание, разумно соблюдать пост и учиться благодарности вместо пресыщения",
                "old_testament": ("Не будь между упивающимися вином, между пресыщающимися мясом.", "Притч. 23:20-21"),
                "apostle": (
                    "Их бог - чрево, и слава их в сраме; они мыслят о земном.",
                    "Флп. 3:19",
                ),
                "father": ("Невоздержание тела ослабляет волю и делает трудной молитвенную жизнь.", "Прп. Иоанн Лествичник"),
                "preacher": ("Пост возвращает человеку свободу от диктата привычек и учит благодарить Бога.", "Протопресвитер Александр Шмеман"),
            },
            "despondency": {
                "title_topic": "Грех уныния",
                "name_genitive": "греха уныния",
                "keywords": ["унын", "отчаян", "леност", "праздност", "надежд", "бодрств"],
                "focus": "он лишает человека мужества и внушает мысль, будто исправление уже невозможно",
                "practice": "сохранять молитвенное правило, не оставлять добрые дела и держаться церковной жизни даже в немощи",
                "old_testament": ("Веселое сердце благотворно, как врачевство, а унылый дух сушит кости.", "Притч. 17:22"),
                "apostle": ("Все заботы ваши возложите на Него, ибо Он печется о вас.", "1 Пет. 5:7"),
                "father": ("Уныние побеждается терпением, молитвой и верностью в малом труде.", "Прп. Ефрем Сирин"),
                "preacher": ("Надежда рождается там, где человек, несмотря на тьму, продолжает стоять перед Богом.", "Митрополит Антоний Сурожский"),
            },
            "judgment": {
                "title_topic": "Грех осуждения",
                "name_genitive": "греха осуждения",
                "keywords": ["осужд", "злослов", "клевет", "язык", "милост", "смирен"],
                "focus": "он разрушает любовь и незаметно возносит человека над ближним",
                "practice": "хранить язык, молиться за ближнего и сначала судить собственную совесть",
                "old_testament": ("Положи, Господи, охрану устам моим, и огради двери уст моих.", "Пс. 140:3"),
                "apostle": (
                    "Не злословьте друг друга, братия... Един Законодатель и Судия, могущий спасти и погубить; а ты кто, который судишь другого?",
                    "Иак. 4:11-12",
                ),
                "father": ("Кто осуждает брата, тот умножает собственные раны и теряет мир сердца.", "Прп. Авва Дорофей"),
                "preacher": ("Осуждение лечится памятью о своих грехах и состраданием к немощи другого.", "Свт. Тихон Задонский"),
            },
            "lying": {
                "title_topic": "Грех лжи",
                "name_genitive": "греха лжи",
                "keywords": ["лжи", "лож", "обман", "лукав", "истин", "правд"],
                "focus": "она разрушает доверие, искажает совесть и делает молитву лицемерной",
                "practice": "говорить правду с любовью, каяться в лукавстве и восстанавливать доверие делами",
                "old_testament": ("Мерзость пред Господом - уста лживые, а говорящие истину благоугодны Ему.", "Притч. 12:22"),
                "apostle": (
                    "Отвергнув ложь, говорите истину каждый ближнему своему, потому что мы члены друг другу.",
                    "Еф. 4:25",
                ),
                "father": ("Ложь делает душу двоедушной и лишает человека внутренней целостности.", "Свт. Василий Великий"),
                "preacher": ("Там, где нет правды в слове, там гаснет и правда в сердце.", "Свт. Иоанн Кронштадтский"),
            },
            "generic_sin": {
                "title_topic": "Грех и покаяние",
                "name_genitive": "греха и нераскаянной страсти",
                "keywords": ["грех", "страст", "покаян", "исповед", "исправл", "трезвен"],
                "focus": "он отделяет человека от Бога и постепенно лишает сердце мира и любви",
                "practice": "честно исповедовать падения, отвергать самооправдание и жить в ежедневном покаянном труде",
                "old_testament": ("Скрывающий свои преступления не будет иметь успеха, а кто сознается и оставляет их, тот будет помилован.", "Притч. 28:13"),
                "apostle": ("Возмездие за грех - смерть, а дар Божий - жизнь вечная во Христе Иисусе, Господе нашем.", "Рим. 6:23"),
                "father": ("Не столько страшно пасть, сколько, пав, оставаться без покаяния.", "Свт. Иоанн Златоуст"),
                "preacher": ("Покаяние начинается там, где человек перестает оправдывать грех и берет ответственность перед Богом.", "Свт. Феофан Затворник"),
            },
        }
        return profiles.get(code)

    def _is_feast_topic(self, topic_low: str) -> bool:
        low = (topic_low or "").lower()
        if re.search(r"\bвход\w*\s+господ\w*", low) or ("иерусалим" in low and "вход" in low):
            return True
        return any(
            w in topic_low
            for w in [
                "рождеств",
                "благовещ",
                "сретен",
                "крещен",
                "богоявлен",
                "преображ",
                "вознес",
                "успен",
                "вход господ",
                "вербн",
                "троиц",
                "пятидесят",
                "воздвижен",
                "введени",
                "праздник",
                "дванадесят",
                "двунадесят",
            ]
        )

    def _feast_subtopic(self, topic_low: str) -> str:
        low = (topic_low or "").lower()
        if re.search(r"рождеств\w+.*богород", low) or any(
            w in low for w in ["рождеств пресвят", "рождества богород"]
        ):
            return "nativity_theotokos"
        if any(w in low for w in ["воздвижен", "креста господ", "крестовоздвиж"]):
            return "cross_exaltation"
        if (re.search(r"введен\w+.*богород", low) or any(w in low for w in ["введени", "введения", "введенье"])) and "богород" in low:
            return "entry_theotokos"
        if re.search(r"рождеств\w+.*христ", low) or any(w in low for w in ["рождеств христ", "рождества христ"]):
            return "nativity_christ"
        if any(w in low for w in ["крещен", "богоявлен"]):
            return "theophany"
        if any(w in low for w in ["сретен"]):
            return "meeting"
        if any(w in low for w in ["благовещ"]):
            return "annunciation"
        if re.search(r"\bвход\w*\s+господ\w*", low) or any(
            w in low for w in ["иерусалим", "вербн", "ваий"]
        ):
            return "entry_jerusalem"
        if any(w in low for w in ["вознес"]):
            return "ascension"
        if any(w in low for w in ["троиц", "пятидесят", "святого духа", "сошеств"]):
            return "trinity"
        if any(w in low for w in ["преображ"]):
            return "transfiguration"
        if any(w in low for w in ["успен"]):
            return "dormition"
        return "generic_feast"

    def _major_gospel_subtopic(self, topic_low: str) -> str:
        low = (topic_low or "").lower()
        if self._is_marriage_topic_low(low):
            return "wedding_sacrament"
        if any(w in low for w in ["блудн", "притч о блудн", "лук 15"]):
            return "prodigal_son"
        if re.search(r"(добр\w+|милосерд\w+)\s+самарян\w+", low):
            return "good_samaritan"
        if re.search(r"тайн\w+\s+вечер\w+", low) or any(w in low for w in ["тайн вечер", "тайной вечери", "тайной вечере"]):
            return "last_supper"
        if any(w in low for w in ["распят", "голгоф", "крестны страдан", "страстях христ"]):
            return "crucifixion"
        if re.search(r"нагорн\w+\s+проповед\w+", low) or any(
            w in low for w in ["блаженств", "заповедях блаженств"]
        ):
            return "mount_sermon"
        if re.search(r"брак\w*\s+в\s+кан\w+", low) or any(w in low for w in ["кане галилейск"]):
            return "cana"
        if any(w in low for w in ["насыщен", "пять тысяч", "пяти тысяч"]):
            return "multiplication_loaves"
        if any(w in low for w in ["слепорожден", "исцелен слеп", "исцелении слеп"]):
            return "healing_blind"
        if any(w in low for w in ["закхе", "закхей"]):
            return "zacchaeus"
        return ""

    def _event_profile(self, topic_low: str) -> Optional[Dict[str, object]]:
        code = self._feast_subtopic(topic_low)
        if code == "generic_feast":
            code = self._major_gospel_subtopic(topic_low)
        if not code:
            return None

        profiles: Dict[str, Dict[str, object]] = {
            "wedding_sacrament": {
                "name": "Таинства Венчания",
                "keywords": ["венчан", "таинств", "брак", "супруг", "муж", "жен", "семейн", "верност"],
                "focus": "церковный брак является таинственным союзом во Христе, призванным к взаимной любви, верности и совместному пути ко спасению",
                "practice": "беречь супружескую верность, молиться вместе, учиться прощению и нести ответственность друг за друга перед Богом",
                "bible": ("И будут два одна плоть; так что они уже не двое, но одна плоть.", "Мф. 19:5-6"),
                "father": (
                    "Супружество есть таинство любви, где муж и жена становятся соработниками Божией благодати.",
                    "Свт. Иоанн Златоуст",
                ),
                "preacher": (
                    "Христианская семья крепнет там, где супруги вместе молятся, прощают и служат друг другу.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            },
            "nativity_theotokos": {
                "name": "Рождества Пресвятой Богородицы",
                "keywords": ["рождеств", "богород", "мария", "смирен", "надежд"],
                "focus": "через смирение и чистоту Бог готовит миру пришествие Спасителя",
                "practice": "учиться кротости, благодарению и верности Богу в повседневных делах",
                "bible": ("Се, Раба Господня; да будет Мне по слову твоему.", "Лк. 1:38"),
                "father": (
                    "Через Пречистую Деву миру явилась надежда спасения и начало новой жизни во Христе.",
                    "Свт. Григорий Палама",
                ),
                "preacher": (
                    "Праздник Богородицы зовет нас к доверию Богу и тишине сердца.",
                    "Протопресвитер Александр Шмеман",
                ),
            },
            "cross_exaltation": {
                "name": "Воздвижения Креста Господня",
                "keywords": ["воздвиж", "крест", "голгоф", "жертв", "смирен"],
                "focus": "Крест Христов являет победу любви Божией над грехом и смертью",
                "practice": "нести свой крест с терпением, не роптать и хранить верность заповедям",
                "bible": ("Кто хочет идти за Мною, отвергнись себя, и возьми крест свой, и следуй за Мною.", "Мк. 8:34"),
                "father": ("Крест есть слава Христова и путь нашего спасения.", "Свт. Иоанн Златоуст"),
                "preacher": (
                    "Почитание Креста начинается там, где человек учится жертвенной любви к ближним.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            },
            "entry_theotokos": {
                "name": "Введения во храм Пресвятой Богородицы",
                "keywords": ["введени", "храм", "богород", "посвящен", "молитв"],
                "focus": "человек с детства призван посвящать себя Богу и хранить чистоту сердца",
                "practice": "воспитывать веру в семье, беречь молитвенный уклад и церковность жизни",
                "bible": ("Блаженны чистые сердцем, ибо они Бога узрят.", "Мф. 5:8"),
                "father": ("Храм Божий прежде всего должен созидаться в сердце человека.", "Свт. Феофан Затворник"),
                "preacher": (
                    "Введение Богородицы напоминает нам, что духовная жизнь начинается с верности в малом.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "nativity_christ": {
                "name": "Рождества Христова",
                "keywords": ["рождеств", "христ", "воплощен", "вифлеем", "радост"],
                "focus": "Бог становится Человеком ради спасения мира и освящения человеческой жизни",
                "practice": "жить в благодарении, мире с ближними и деятельной любви",
                "bible": ("И Слово стало плотию и обитало с нами.", "Ин. 1:14"),
                "father": ("Бог стал человеком, чтобы человек стал причастником Божественной жизни.", "Свт. Афанасий Великий"),
                "preacher": (
                    "Рождество подлинно тогда, когда Христос рождается в глубине нашего сердца.",
                    "Протоиерей Александр Мень",
                ),
            },
            "theophany": {
                "name": "Крещения Господня (Богоявления)",
                "keywords": ["крещен", "богоявлен", "иордан", "освящен", "обновлен"],
                "focus": "на Иордане открывается тайна Святой Троицы и освящение человеческой природы",
                "practice": "обновлять в себе благодать крещения через покаяние, молитву и чистую жизнь",
                "bible": ("Сей есть Сын Мой возлюбленный, в Котором Мое благоволение.", "Мф. 3:17"),
                "father": ("Христос входит в воды Иордана, чтобы освятить весь мир и человека.", "Свт. Григорий Богослов"),
                "preacher": (
                    "Праздник Богоявления зовет нас вспомнить о своем крещальном обещании верности Христу.",
                    "Свт. Тихон Задонский",
                ),
            },
            "meeting": {
                "name": "Сретения Господня",
                "keywords": ["сретен", "встреч", "симеон", "храм", "ожидан"],
                "focus": "человек призван встретить Христа и держать сердце открытым для Его пришествия",
                "practice": "жить в бодрствовании, молитве и терпеливом ожидании воли Божией",
                "bible": ("Ныне отпускаешь раба Твоего, Владыко, по слову Твоему, с миром.", "Лк. 2:29"),
                "father": ("Сретение учит нас, что истинная встреча с Богом требует чистоты сердца.", "Свт. Амвросий Медиоланский"),
                "preacher": (
                    "Как праведный Симеон, будем ждать Христа не в страхе, а в надежде.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "annunciation": {
                "name": "Благовещения Пресвятой Богородицы",
                "keywords": ["благовещ", "богород", "послушан", "смирен", "благодат"],
                "focus": "послушание Божией воле открывает путь спасения и подлинной свободы человека",
                "practice": "учиться говорить Богу «да», отвергая гордость и своеволие",
                "bible": ("Се, Раба Господня; да будет Мне по слову твоему.", "Лк. 1:38"),
                "father": ("Через согласие Девы Марии человечество вновь получает надежду на спасение.", "Свт. Иоанн Дамаскин"),
                "preacher": (
                    "Благовещение учит нас доверять Богу даже тогда, когда путь еще не ясен.",
                    "Протопресвитер Александр Шмеман",
                ),
            },
            "entry_jerusalem": {
                "name": "Входа Господня в Иерусалим",
                "keywords": ["вход", "иерусалим", "верб", "осанна", "смирен"],
                "focus": "Христос как Царь смирения зовет нас к верности не только в радости, но и в испытании",
                "practice": "встречать Господа покаянием, молитвой и внутренней собранностью",
                "bible": ("Осанна Сыну Давидову! Благословен Грядущий во имя Господне!", "Мф. 21:9"),
                "father": ("Вход Господень напоминает, что Христос приходит к смиренным сердцам.", "Свт. Феофан Затворник"),
                "preacher": (
                    "Вербное воскресенье призывает нас встречать Христа не внешностью, а жизнью.",
                    "Протоиерей Александр Мень",
                ),
            },
            "ascension": {
                "name": "Вознесения Господня",
                "keywords": ["вознес", "неб", "христ", "надежд", "церков"],
                "focus": "Вознесение открывает человеку небесное призвание и уверенность в Божием попечении",
                "practice": "жить с надеждой, храня верность Церкви и заповедям в земных трудах",
                "bible": ("И когда благословлял их, стал отдаляться от них и возноситься на небо.", "Лк. 24:51"),
                "father": ("Вознесение Господне возводит ум христианина от земного к небесному.", "Свт. Лев Великий"),
                "preacher": (
                    "Праздник Вознесения учит нас не терять духовной высоты среди ежедневной суеты.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            },
            "trinity": {
                "name": "Дня Святой Троицы (Пятидесятницы)",
                "keywords": ["троиц", "пятидесят", "дух свят", "благодат", "единств"],
                "focus": "Церковь живет силой Духа Святого, Который собирает нас в единство любви",
                "practice": "просить просвещения Духа Святого и хранить мир, кротость и братолюбие",
                "bible": ("И исполнились все Духа Святаго и начали говорить на иных языках.", "Деян. 2:4"),
                "father": ("Дух Святой не разделяет, но собирает Церковь в любви и истине.", "Свт. Василий Великий"),
                "preacher": (
                    "Пятидесятница призывает нас к жизни, открытой действию благодати в каждом дне.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "transfiguration": {
                "name": "Преображения Господня",
                "keywords": ["преображ", "фавор", "свет", "обновлен", "христ"],
                "focus": "Фаворский свет показывает цель жизни человека - преображение во Христе",
                "practice": "очищать сердце покаянием и терпеливо восходить от страстей к свету Божию",
                "bible": ("И преобразился пред ними: и просияло лице Его, как солнце.", "Мф. 17:2"),
                "father": ("Преображение Господне открывает славу Божию и призвание человека к обожению.", "Свт. Григорий Палама"),
                "preacher": (
                    "Путь к Фавору начинается в верности молитве и борьбе со страстями.",
                    "Свт. Феофан Затворник",
                ),
            },
            "dormition": {
                "name": "Успения Пресвятой Богородицы",
                "keywords": ["успен", "богород", "надежд", "мир", "вечн"],
                "focus": "Успение учит нас христианской надежде и мирному преданию себя в волю Божию",
                "practice": "жить в покаянии и благодарении, готовя сердце к встрече с Господом",
                "bible": ("Блаженны мертвые, умирающие в Господе.", "Откр. 14:13"),
                "father": ("В Успении Богородицы Церковь видит торжество жизни над тлением.", "Свт. Иоанн Дамаскин"),
                "preacher": (
                    "Праздник Успения учит нас не страху смерти, а доверию Богу и надежде воскресения.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "good_samaritan": {
                "name": "притчи о милосердном самарянине",
                "keywords": ["самарян", "милосерд", "ближн", "сострадан", "помощ"],
                "focus": "любовь к ближнему проверяется делом милосердия, а не словами",
                "practice": "не проходить мимо чужой боли, быть внимательными и деятельными в любви",
                "bible": ("Иди, и ты поступай так же.", "Лк. 10:37"),
                "father": ("Милосердие есть подлинный признак евангельской жизни.", "Свт. Иоанн Златоуст"),
                "preacher": (
                    "Ближний - это тот, кому ты сегодня можешь послужить любовью.",
                    "Протоиерей Александр Мень",
                ),
            },
            "last_supper": {
                "name": "Тайной Вечери",
                "keywords": ["тайн", "вечер", "чаша", "евхарист", "любов"],
                "focus": "Господь дарует Церкви Евхаристию как источник единства и жизни",
                "practice": "готовить сердце к Причастию через покаяние, примирение и благодарение",
                "bible": ("Сие творите в Мое воспоминание.", "Лк. 22:19"),
                "father": ("Евхаристия соединяет нас со Христом и делает нас одним Телом.", "Свт. Кирилл Иерусалимский"),
                "preacher": (
                    "Тайная Вечеря зовет нас к ответственности за церковное единство и чистоту сердца.",
                    "Свт. Иоанн Кронштадтский",
                ),
            },
            "crucifixion": {
                "name": "Крестных страданий Христовых",
                "keywords": ["крест", "голгоф", "страдан", "жертв", "спасен"],
                "focus": "на Кресте открывается глубина Божией любви и цена человеческого спасения",
                "practice": "учиться терпению, прощению и жертвенной любви ко всем людям",
                "bible": ("Отче! прости им, ибо не знают, что делают.", "Лк. 23:34"),
                "father": ("Сила Креста побеждает ожесточение и исцеляет сердце человека.", "Свт. Игнатий (Брянчанинов)"),
                "preacher": (
                    "Созерцая Голгофу, человек учится любить не словом только, но жертвой.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            },
            "mount_sermon": {
                "name": "Нагорной проповеди",
                "keywords": ["нагорн", "блаженств", "заповед", "сердц", "царств"],
                "focus": "заповеди блаженств открывают путь подлинного евангельского счастья",
                "practice": "взращивать кротость, чистоту сердца и жажду правды в повседневной жизни",
                "bible": ("Блаженны чистые сердцем, ибо они Бога узрят.", "Мф. 5:8"),
                "father": ("Блаженства - это не идеал для избранных, а путь каждого христианина.", "Свт. Иоанн Златоуст"),
                "preacher": (
                    "Нагорная проповедь учит нас побеждать зло не силой, а силой любви и правды.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "cana": {
                "name": "брака в Кане Галилейской",
                "keywords": ["кана", "брак", "чудо", "радост", "семь"],
                "focus": "Христос благословляет семейную жизнь и преображает человеческую радость",
                "practice": "строить семью на молитве, взаимном уважении и терпении",
                "bible": ("Так положил Иисус начало чудесам в Кане Галилейской.", "Ин. 2:11"),
                "father": ("Присутствие Христа в семье делает дом местом благодати.", "Свт. Феофан Затворник"),
                "preacher": (
                    "Чудо в Кане напоминает: без Бога человеческой радости не хватает глубины и мира.",
                    "Протоиерей Александр Мень",
                ),
            },
            "multiplication_loaves": {
                "name": "насыщения пяти тысяч",
                "keywords": ["насыщен", "пять тысяч", "хлеб", "чудо", "милосерд"],
                "focus": "Господь питает человека не только хлебом земным, но и словом жизни",
                "practice": "делиться с нуждающимися и учиться доверять Божьему попечению",
                "bible": ("Вы дайте им есть.", "Мф. 14:16"),
                "father": ("Малое в руках Божиих становится великим даром для многих.", "Свт. Иоанн Златоуст"),
                "preacher": (
                    "Чудо насыщения учит нас щедрости и вере, что Господь восполняет недостающее.",
                    "Свт. Иоанн Кронштадтский",
                ),
            },
            "healing_blind": {
                "name": "исцеления слепорожденного",
                "keywords": ["слеп", "исцелен", "свет", "вера", "христ"],
                "focus": "Христос открывает человеку духовное зрение и ведет от тьмы к свету",
                "practice": "просить у Бога просвещения ума и хранить верность истине в испытаниях",
                "bible": ("Я свет миру.", "Ин. 9:5"),
                "father": ("Телесное исцеление указывает на более глубокое исцеление души.", "Свт. Кирилл Александрийский"),
                "preacher": (
                    "Без внутреннего зрения совести человек теряет путь, даже видя внешне.",
                    "Митрополит Антоний Сурожский",
                ),
            },
            "zacchaeus": {
                "name": "обращения Закхея",
                "keywords": ["закхе", "покаян", "обращен", "исправлен", "милосерд"],
                "focus": "истинная встреча со Христом приводит к покаянию и изменению жизни",
                "practice": "исправлять несправедливость, быть щедрыми и честными перед Богом и людьми",
                "bible": ("Ныне пришло спасение дому сему.", "Лк. 19:9"),
                "father": ("Покаяние становится истинным, когда меняет поступки человека.", "Свт. Тихон Задонский"),
                "preacher": (
                    "Закхей показывает, что путь к Богу начинается с решимости встать выше прежней жизни.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            },
        }
        return profiles.get(code)

    def _is_saint_topic(self, topic_low: str) -> bool:
        if any(w in topic_low for w in ["богород", "пасх", "воскрес", "лазар"]):
            return False
        return any(
            w in topic_low
            for w in [
                "свят",
                "свт.",
                "преп",
                "мучен",
                "блаженн",
                "праведн",
                "апостол",
                "исповедник",
                "чудотвор",
            ]
        )

    def _is_marriage_topic_low(self, topic_low: str) -> bool:
        low = (topic_low or "").lower()
        if any(
            w in low
            for w in [
                "таинств венчан",
                "таинство венчан",
                "венчан",
                "церковн брак",
                "таинство брака",
                "супруж",
                "супруг",
                "муж и жен",
            ]
        ):
            return True
        if "брак" in low and any(w in low for w in ["христиан", "православ", "семейн", "муж", "жен"]):
            return True
        return False

    def _topic_specific_keywords(self, topic: str) -> List[str]:
        low = self.preprocessor.normalize(topic or "").lower()
        if not low:
            return []
        if self._is_marriage_topic_low(low):
            return ["венчан", "таинств", "брак", "супруг", "муж", "жен", "семейн", "верност"]
        stop = {
            "проповедь",
            "подготовь",
            "подготовить",
            "сгенерируй",
            "составь",
            "напиши",
            "тема",
            "теме",
            "тему",
            "православной",
            "православную",
            "православная",
            "святой",
            "святого",
            "святых",
            "день",
            "дне",
            "жизни",
            "жизнь",
            "христианина",
            "христианской",
            "христианина",
            "праздник",
            "действии",
            "благодати",
            "жизнь",
            "человека",
            "церкви",
            "церковь",
            "бог",
            "бога",
            "христос",
            "христа",
            "господь",
            "господа",
        }
        words = [w for w in re.findall(r"[а-яёa-z]{4,}", low) if w not in stop]
        markers: List[str] = []
        for w in words:
            m = w[: max(4, len(w) - 1)]
            if m not in markers:
                markers.append(m)
        return markers[:8]

    def _ensure_paschal_conclusion(self, sermon: str, req: GenerateRequest) -> str:
        if not self._is_resurrection_topic(req):
            return sermon
        intro, main, concl = self._split_sermon_sections(sermon)
        if not concl:
            return sermon
        concl_low = concl.lower()
        if "христос воскресе" not in concl_low:
            concl = concl.rstrip()
            if concl and not concl.endswith((".", "!", "?")):
                concl += "."
            concl += " Христос Воскресе, дорогие братья и сестры! Воистину Воскресе!"
            title = self._compose_title(req)
            sermon = (
                f"{title}\n\n"
                f"Вступление.\n{intro}\n\n"
                f"Основная часть.\n{main}\n\n"
                f"Заключение.\n{concl}"
            )
        return sermon

    def _ensure_amen_last(self, sermon: str, req: GenerateRequest) -> str:
        intro, main, concl = self._split_sermon_sections(sermon)
        if not concl:
            return sermon
        clean_concl = self.preprocessor.normalize(concl)
        # Убираем все вхождения «Аминь» внутри заключения и ставим одно в самый конец.
        clean_concl = re.sub(r"\bаминь\.?\b", " ", clean_concl, flags=re.IGNORECASE)
        clean_concl = re.sub(r"\s+", " ", clean_concl).strip()
        if clean_concl and clean_concl[-1] not in ".!?":
            clean_concl += "."
        clean_concl = (clean_concl + " Аминь.").strip() if clean_concl else "Аминь."
        title = self._compose_title(req)
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{clean_concl}"
        )

    def _topic_markers(self, topic: str) -> List[str]:
        low = self.preprocessor.normalize(topic).lower()
        if not low:
            return []

        event_profile = self._event_profile(low)
        if event_profile:
            kws = [str(x) for x in event_profile.get("keywords", [])]
            return kws[:8]
        sin_profile = self._sin_profile(low)
        if sin_profile:
            return [str(x) for x in sin_profile.get("keywords", [])][:10]
        if any(w in low for w in ["лазар", "лазарев", "лазарева суббот", "четвероднев", "вифан"]):
            return ["лазар", "лазарев", "вифан", "четвероднев", "марф", "мария", "суббот"]
        if any(w in low for w in ["блудн", "притч о блудн", "лук 15", "расточил имение"]):
            return ["блудн", "наслед", "покаян", "возврат", "отец", "старш", "младш"]
        if self._is_feast_topic(low):
            return [
                "праздник",
                "церковь",
                "христ",
                "бог",
                "благодат",
                "спасен",
            ]
        if self._is_saint_topic(low):
            return ["свят", "жити", "подвиг", "смирен", "молитв", "добродетел"]
        if any(w in low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            return ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]
        if self._is_marriage_topic_low(low):
            return ["венчан", "таинств", "брак", "супруг", "супруж", "муж", "жен", "семейн", "верност"]
        if self._is_sin_topic_low(low):
            return ["грех", "страст", "покаян", "исповед", "исправл", "воздерж", "совест"]
        if self._is_paschal_topic_low(low):
            return ["воскрес", "христос воскрес", "пасх"]

        stop = {
            "проповедь",
            "подготовь",
            "подготовить",
            "сгенерируй",
            "тема",
            "теме",
            "тему",
            "о",
            "про",
            "на",
            "и",
            "для",
            "об",
            "по",
        }
        words = [w for w in re.findall(r"[а-яёa-z]{4,}", low) if w not in stop]
        markers = []
        for w in words:
            marker = w[: max(4, len(w) - 2)]
            if marker not in markers:
                markers.append(marker)
        return markers[:6]

    def _topic_is_covered(self, text: str, req: GenerateRequest) -> bool:
        topic = self._extract_topic(req)
        markers = self._topic_markers(topic)
        if not markers:
            return True
        low = (text or "").lower()
        _, main_section, _ = self._split_sermon_sections(text)
        main_low = (main_section or low).lower()
        strict_lock = self._topic_lock_is_strict(req)
        coverage_low = main_low if strict_lock else low
        topic_low = topic.lower()
        event_profile = self._event_profile(topic_low)
        sin_profile = self._sin_profile(topic_low)
        hits = sum(1 for m in markers if m in coverage_low)
        topic_words = [
            w
            for w in re.findall(r"[а-яёa-z]{4,}", topic_low)
            if w not in {"проповедь", "тема", "теме", "тему", "о", "про", "на", "и", "для", "об", "по"}
        ]
        has_topic_name = (
            any((w[: max(4, len(w) - 1)] in coverage_low) for w in topic_words[:5]) if topic_words else True
        )
        if event_profile:
            kws = [str(x) for x in event_profile.get("keywords", [])]
            if self._is_marriage_topic_low(topic_low):
                kw_hits_main = sum(1 for k in kws if k in main_low)
                spouse_hits = sum(1 for m in ["супруг", "супруж", "муж", "жен"] if m in main_low)
                sacr_hits = sum(1 for m in ["венчан", "таинств", "брак"] if m in main_low)
                return kw_hits_main >= 3 and spouse_hits >= 1 and sacr_hits >= 1
            kw_hits = sum(1 for k in kws if k in coverage_low)
            return kw_hits >= 2 or (kw_hits >= 1 and has_topic_name)
        if sin_profile:
            kws = [str(x) for x in sin_profile.get("keywords", [])]
            kw_hits = sum(1 for k in kws if k in coverage_low)
            repentance_hits = sum(
                1 for m in ["покаян", "исповед", "исправл", "отреч", "трезвен", "молитв"] if m in coverage_low
            )
            return kw_hits >= 2 and repentance_hits >= 1
        if "лазар" in markers:
            lazarus_hits = sum(
                1 for m in ["лазар", "лазарев", "вифан", "четвероднев", "марф", "мария"] if m in coverage_low
            )
            return lazarus_hits >= 2 or ("лазар" in coverage_low and "суббот" in coverage_low)
        if "блудн" in markers:
            prodigal_hits = sum(1 for m in ["блудн", "покаян", "наслед", "возврат", "отец"] if m in coverage_low)
            return prodigal_hits >= 2 and ("блудн" in coverage_low or "притч" in coverage_low)
        if self._is_saint_topic(topic_low):
            return has_topic_name and any(m in coverage_low for m in ["свят", "подвиг", "добродетел", "молитв"])
        if self._is_feast_topic(topic_low):
            return has_topic_name and any(m in coverage_low for m in ["праздник", "церковь", "господ", "благодат"])
        if self._is_sin_topic_low(topic_low):
            sin_hits = sum(
                1 for m in ["грех", "страст", "паден", "порок", "беззакони", "страх бож"] if m in coverage_low
            )
            repentance_hits = sum(
                1
                for m in ["покаян", "исповед", "исправл", "воздерж", "трезвен", "молитв", "борьб"]
                if m in coverage_low
            )
            return sin_hits >= 1 and repentance_hits >= 1
        if "богород" in markers:
            # Для Богородичной темы требуем явное попадание в богородичную лексику.
            return any(m in coverage_low for m in ["богород", "пресвят", "дева мар", "матер бож"])
        if "воскрес" in markers:
            # Для пасхальной темы требуем явный акцент именно на Воскресении, а не только на слове "Пасха".
            return coverage_low.count("воскрес") >= 2 or "христос воскресе" in coverage_low
        return hits >= 1

    def _source_kind(self, source_type: str) -> str:
        low = (source_type or "").lower()
        if "bible" in low or "писан" in low:
            return "bible"
        if "comment" in low or "толк" in low:
            return "father"
        if "sermon" in low or "пропов" in low:
            return "preacher"
        return "other"

    def _citation_relevance_score(self, citation: Citation, markers: List[str]) -> int:
        text = " ".join(
            [
                citation.excerpt or "",
                citation.title or "",
                citation.reference or "",
                citation.author or "",
            ]
        ).lower()
        hits = sum(text.count(m) for m in markers if m)
        score = int(round(citation.score * 1000))
        return score + hits * 50

    def _extract_quote_sentence(
        self,
        text: str,
        markers: List[str],
        require_marker: bool = False,
        specific_markers: Optional[List[str]] = None,
    ) -> Optional[str]:
        clean = self.preprocessor.normalize(text or "")
        clean = re.sub(r"https?://\S+", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()
        if not clean:
            return None

        candidates = self.preprocessor.split_into_sentences(clean)
        if not candidates:
            candidates = [clean]

        def is_good(sentence: str) -> bool:
            low = sentence.lower()
            words = re.findall(r"[А-Яа-яA-Za-zЁё]+", sentence)
            if len(words) < 5 or len(words) > 44:
                return False
            if len(sentence) > 360:
                return False
            if "http" in low or "royallib" in low or "livejournal" in low or "mso-" in low:
                return False
            if re.search(r"\d", sentence):
                return False
            if "[" in sentence or "]" in sentence:
                return False
            if "(" in sentence or ")" in sentence:
                return False
            if low.count(";") >= 3:
                return False
            if sentence.strip().startswith(("—", "-", "•")):
                return False
            return True

        marker_hits: List[str] = []
        fallback_hits: List[str] = []
        for s in candidates:
            s_norm = s.strip(" -—\n\t")
            if not s_norm:
                continue
            if not is_good(s_norm):
                continue
            low = s_norm.lower()
            marker_hit = any(m in low for m in markers)
            if marker_hit and specific_markers:
                marker_hit = any(m in low for m in specific_markers)
            if marker_hit:
                marker_hits.append(s_norm)
            else:
                fallback_hits.append(s_norm)

        if require_marker and not marker_hits:
            return None

        picked = marker_hits[0] if marker_hits else (fallback_hits[0] if fallback_hits else None)
        if not picked:
            return None
        picked = picked.strip()
        if picked and picked[-1] not in ".!?":
            picked += "."
        return picked

    def _sin_quote_extra_pool(self, code: str) -> Dict[str, List[Tuple[str, str]]]:
        generic = {
            "old_testament": [
                ("Сердце чистое сотвори во мне, Боже, и дух правый обнови внутри меня.", "Пс. 50:12"),
                ("Омойтесь, очиститесь; удалите злые деяния ваши от очей Моих; перестаньте делать зло.", "Ис. 1:16"),
            ],
            "apostle": [
                ("Испытывайте самих себя, в вере ли вы; самих себя исследывайте.", "2 Кор. 13:5"),
                ("Итак покайтесь и обратитесь, чтобы загладились грехи ваши.", "Деян. 3:19"),
            ],
            "father": [
                ("Покаяние есть второе крещение и возвращение души к миру с Богом.", "Свт. Иоанн Златоуст"),
                ("Кто хранит внимание к совести, тот быстрее замечает начало страсти.", "Свт. Феофан Затворник"),
            ],
            "preacher": [
                ("Господь ждет от нас не объяснений, а перемены сердца и честного труда покаяния.", "Митрополит Антоний Сурожский"),
                ("Победа над страстью начинается с малого: молитва, трезвение и верность в ежедневном делании.", "Свт. Лука (Войно-Ясенецкий)"),
            ],
        }

        specific: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
            "judgment": {
                "old_testament": [
                    ("Кто хранит уста свои и язык свой, тот хранит от бед душу свою.", "Притч. 21:23"),
                    ("Не ходи переносчиком в народе твоем и не восставай на жизнь ближнего твоего.", "Лев. 19:16"),
                ],
                "apostle": [
                    ("Не судите никак прежде времени, пока не придет Господь.", "1 Кор. 4:5"),
                    ("Кто ты, осуждающий чужого раба? Перед своим Господом стоит он, или падает.", "Рим. 14:4"),
                ],
                "father": [
                    ("Осуждение ближнего показывает, что человек забыл о собственных грехах.", "Прп. Авва Дорофей"),
                    ("Ничто так не удаляет благодать, как привычка судить другого.", "Свт. Иоанн Златоуст"),
                ],
                "preacher": [
                    ("Пока мы заняты чужими ошибками, мы теряем время для собственного покаяния.", "Свт. Тихон Задонский"),
                    ("Осуждение всегда начинается с гордости и заканчивается внутренней пустотой.", "Протопресвитер Александр Шмеман"),
                ],
            },
            "divination": {
                "old_testament": [
                    ("Не обращайтесь к вызывающим мертвых и к волшебникам; не доводите себя до осквернения от них.", "Лев. 19:31"),
                    ("Истреблен будет народ Мой за недостаток ведения.", "Ос. 4:6"),
                ],
                "apostle": [
                    ("Что общего у света со тьмою? ... выйдите из среды их и отделитесь.", "2 Кор. 6:14-17"),
                    ("Бодрствуйте, стойте в вере, будьте мужественны, тверды.", "1 Кор. 16:13"),
                ],
            },
            "pride": {
                "old_testament": [
                    ("Смиренных возносит Господь, а нечестивых унижает до земли.", "Пс. 146:6"),
                    ("Перед падением возносится сердце человека, а смирение предшествует славе.", "Притч. 18:13"),
                ],
                "apostle": [
                    ("Ничего не делайте по любопрению или по тщеславию, но по смиренномудрию почитайте один другого выше себя.", "Флп. 2:3"),
                    ("Облекитесь смиренномудрием, потому что Бог гордым противится, а смиренным дает благодать.", "1 Пет. 5:5"),
                ],
            },
            "anger": {
                "old_testament": [
                    ("Кроткий ответ отвращает гнев, а оскорбительное слово возбуждает ярость.", "Притч. 15:1"),
                    ("Перестань гневаться и оставь ярость; не ревнуй до того, чтобы делать зло.", "Пс. 36:8"),
                ],
                "apostle": [
                    ("Солнце да не зайдет во гневе вашем; и не давайте места диаволу.", "Еф. 4:26-27"),
                    ("Всякое раздражение и ярость, и гнев ... да будут удалены от вас.", "Еф. 4:31"),
                ],
            },
            "lust": {
                "old_testament": [
                    ("Как юноше содержать в чистоте путь свой? - хранением себя по слову Твоему.", "Пс. 118:9"),
                    ("Завет положил я с глазами моими, чтобы не помышлять мне о девице.", "Иов. 31:1"),
                ],
                "apostle": [
                    ("Бегайте блуда. Всякий грех ... а блудник грешит против собственного тела.", "1 Кор. 6:18"),
                    ("Умертвите земные члены ваши: блуд, нечистоту, страсть, злую похоть.", "Кол. 3:5"),
                ],
            },
            "greed": {
                "old_testament": [
                    ("Не отказывай в благодеянии нуждающемуся, когда рука твоя в силе сделать его.", "Притч. 3:27"),
                    ("Лучше немногое у праведника, нежели изобилие у многих нечестивых.", "Пс. 36:16"),
                ],
                "apostle": [
                    ("Имейте нрав несребролюбивый, довольствуясь тем, что есть.", "Евр. 13:5"),
                    ("Наставляй... чтобы они благодетельствовали, богатели добрыми делами, были щедры и общительны.", "1 Тим. 6:17-18"),
                ],
            },
            "envy": {
                "old_testament": [
                    ("Не ревнуй злодеям и не завидуй делающим беззаконие.", "Пс. 36:1"),
                    ("Кроткое сердце - жизнь для тела, а зависть - гниль для костей.", "Притч. 14:30"),
                ],
                "apostle": [
                    ("Любовь не завидует, любовь не превозносится, не гордится.", "1 Кор. 13:4"),
                    ("Не будем тщеславиться, друг друга раздражать, друг другу завидовать.", "Гал. 5:26"),
                ],
            },
            "drunkenness": {
                "old_testament": [
                    ("Не будь между упивающимися вином, между пресыщающимися мясом.", "Притч. 23:20"),
                    ("Горе тем, которые с раннего утра ищут сикеры.", "Ис. 5:11"),
                ],
                "apostle": [
                    ("Будем бодрствовать и трезвиться... облекшись в броню веры и любви.", "1 Фес. 5:6-8"),
                    ("Трезвитесь, бодрствуйте, потому что противник ваш диавол ходит, как рыкающий лев.", "1 Пет. 5:8"),
                ],
            },
            "gluttony": {
                "old_testament": [
                    ("Не пресыщайся всякою сладостью и не бросайся на разные яства.", "Сир. 37:32"),
                    ("Положи нож к горлу твоему, если ты алчен.", "Притч. 23:2"),
                ],
                "apostle": [
                    ("Но усмиряю и порабощаю тело мое, дабы, проповедуя другим, самому не остаться недостойным.", "1 Кор. 9:27"),
                    ("Ибо Царствие Божие не пища и питие, но праведность и мир и радость во Святом Духе.", "Рим. 14:17"),
                ],
            },
            "despondency": {
                "old_testament": [
                    ("Что унываешь ты, душа моя, и что смущаешься? Уповай на Бога.", "Пс. 41:6"),
                    ("Крепитесь, и да укрепляется сердце ваше, все надеющиеся на Господа.", "Пс. 30:25"),
                ],
                "apostle": [
                    ("Будьте постоянны в молитве, бодрствуя в ней с благодарением.", "Кол. 4:2"),
                    ("Бог верен, Который не попустит вам быть искушаемыми сверх сил.", "1 Кор. 10:13"),
                ],
            },
            "lying": {
                "old_testament": [
                    ("Удаляй от меня путь лжи и закон Твой даруй мне.", "Пс. 118:29"),
                    ("Лживый язык ненавидит уязвляемых им, и льстивые уста готовят падение.", "Притч. 26:28"),
                ],
                "apostle": [
                    ("Не лгите друг другу, совлекшись ветхого человека с делами его.", "Кол. 3:9"),
                    ("Посему, отвергнув ложь, говорите истину каждый ближнему своему.", "Еф. 4:25"),
                ],
            },
        }

        out: Dict[str, List[Tuple[str, str]]] = {}
        spec = specific.get(code, {})
        for kind in ["old_testament", "apostle", "father", "preacher"]:
            out[kind] = list(spec.get(kind, [])) + list(generic.get(kind, []))
        return out

    def _quote_bank_alternatives(self, req: GenerateRequest) -> Dict[str, List[Tuple[str, str]]]:
        topic_low = self._extract_topic(req).lower()
        sin_profile = self._sin_profile(topic_low)
        if sin_profile:
            sin_code = self._sin_subtopic(topic_low)
            extra_pool = self._sin_quote_extra_pool(sin_code)
            old_testament = tuple(
                sin_profile.get(
                    "old_testament",
                    (
                        "Скрывающий свои преступления не будет иметь успеха, а кто сознается и оставляет их, тот будет помилован.",
                        "Притч. 28:13",
                    ),
                )
            )  # type: ignore[arg-type]
            apostle = tuple(
                sin_profile.get(
                    "apostle",
                    (
                        "Отложите прежний образ жизни ветхого человека и обновитесь духом ума вашего.",
                        "Еф. 4:22-23",
                    ),
                )
            )  # type: ignore[arg-type]
            father = tuple(
                sin_profile.get(
                    "father",
                    ("Не столько страшно пасть, сколько, пав, оставаться без покаяния.", "Свт. Иоанн Златоуст"),
                )
            )  # type: ignore[arg-type]
            preacher = tuple(
                sin_profile.get(
                    "preacher",
                    ("Покаяние начинается там, где человек перестает оправдывать грех.", "Свт. Феофан Затворник"),
                )
            )  # type: ignore[arg-type]
            packed = {
                "old_testament": [old_testament] + list(extra_pool.get("old_testament", [])),
                "apostle": [apostle] + list(extra_pool.get("apostle", [])),
                "father": [father] + list(extra_pool.get("father", [])),
                "preacher": [preacher] + list(extra_pool.get("preacher", [])),
            }
            result: Dict[str, List[Tuple[str, str]]] = {}
            for kind, values in packed.items():
                uniq: List[Tuple[str, str]] = []
                seen = set()
                for quote, ref in values:
                    key = (self.preprocessor.normalize(quote).lower(), self.preprocessor.normalize(ref).lower())
                    if key in seen:
                        continue
                    seen.add(key)
                    uniq.append((quote, ref))
                result[kind] = uniq
            return result

        if self._is_lazarus_topic(req):
            return {
                "old_testament": [
                    ("Так говорит Господь Бог костям сим: вот, Я введу дух в вас, и оживете.", "Иез. 37:5"),
                    ("Бог избавит душу мою от власти преисподней, когда примет меня.", "Пс. 48:16"),
                    ("А я знаю, Искупитель мой жив, и Он в последний день восставит из праха распадающуюся кожу мою.", "Иов. 19:25"),
                ],
                "apostle": [
                    ("Если веруем, что Иисус умер и воскрес, то и умерших в Иисусе Бог приведет с Ним.", "1 Фес. 4:14"),
                    ("Как в Адаме все умирают, так во Христе все оживут.", "1 Кор. 15:22"),
                    ("Если Дух Того, Кто воскресил из мертвых Иисуса, живет в вас, то Воскресивший Христа оживит и ваши смертные тела.", "Рим. 8:11"),
                ],
                "bible": [
                    ("Я есмь воскресение и жизнь; верующий в Меня, если и умрет, оживет.", "Ин. 11:25"),
                    ("Лазарь! иди вон.", "Ин. 11:43"),
                    ("Не сказал ли Я тебе, что, если будешь веровать, увидишь славу Божию?", "Ин. 11:40"),
                ],
                "father": [
                    ("Не отчаивайся после падений, но снова и снова начинай покаяние.", "Прп. Иоанн Лествичник"),
                    ("Бог попускает скорби, чтобы пробудить душу к памяти о вечности.", "Свт. Феофан Затворник"),
                    ("Память смертная рождает трезвение и приводит к деятельному покаянию.", "Свт. Игнатий (Брянчанинов)"),
                ],
                "preacher": [
                    ("Когда сердце человека оживает к молитве, он начинает видеть действие Божией благодати во всем.", "Митрополит Антоний Сурожский"),
                    ("Чудо Лазаря напоминает: для Христа нет безнадежных состояний души.", "Свт. Лука (Войно-Ясенецкий)"),
                    ("Вера в воскресение должна менять нашу повседневную жизнь, а не оставаться только знанием.", "Протопресвитер Александр Шмеман"),
                ],
            }
        if self._is_prodigal_topic(req):
            return {
                "old_testament": [
                    ("Разве Я хочу смерти беззаконника? не того ли, чтобы он обратился от путей своих и был жив?", "Иез. 18:23"),
                    ("Раздирайте сердца ваши, а не одежды ваши, и обратитесь к Господу Богу вашему.", "Иоил. 2:13"),
                    ("Семь раз упадет праведник, и встанет.", "Притч. 24:16"),
                ],
                "apostle": [
                    ("Если исповедуем грехи наши, то Он, будучи верен и праведен, простит нам грехи.", "1 Ин. 1:9"),
                    ("Приблизьтесь к Богу, и приблизится к вам.", "Иак. 4:8"),
                    ("Отложите прежний образ жизни ветхого человека и обновитесь духом ума вашего.", "Еф. 4:22-23"),
                ],
                "bible": [
                    ("Встану, пойду к отцу моему и скажу ему: отче! я согрешил против неба и пред тобою.", "Лк. 15:18"),
                    ("Этот сын мой был мертв и ожил, пропадал и нашелся.", "Лк. 15:24"),
                    ("Когда он был еще далеко, увидел его отец его и сжалился.", "Лк. 15:20"),
                ],
                "father": [
                    ("Покаяние не унижает человека, но возвращает ему сыновнее достоинство перед Богом.", "Свт. Иоанн Златоуст"),
                    ("Кто исповедует грех без самооправдания, тот уже начинает исцеляться.", "Свт. Феофан Затворник"),
                    ("Бог приемлет кающегося не по мере слов, а по мере сокрушения сердца.", "Прп. Ефрем Сирин"),
                ],
                "preacher": [
                    ("Бог всегда ждет человека с распростертыми объятиями, но путь домой начинается с решимости встать и идти.", "Митрополит Антоний Сурожский"),
                    ("Возвращение к Богу начинается с честности перед собственной совестью.", "Свт. Лука (Войно-Ясенецкий)"),
                    ("Покаяние становится живым тогда, когда оно меняет отношения с ближними.", "Протоиерей Александр Мень"),
                ],
            }
        if self._is_marriage_topic_low(topic_low):
            return {
                "old_testament": [
                    ("Потому оставит человек отца своего и мать свою и прилепится к жене своей, и будут одна плоть.", "Быт. 2:24"),
                    ("Кто нашел добрую жену, тот нашел благо и получил благодать от Господа.", "Притч. 18:23"),
                    ("Двоим лучше, нежели одному; потому что у них есть доброе вознаграждение в труде их.", "Еккл. 4:9"),
                ],
                "apostle": [
                    ("Мужья, любите своих жен, как и Христос возлюбил Церковь и предал Себя за нее.", "Еф. 5:25"),
                    ("Более же всего облекитесь в любовь, которая есть совокупность совершенства.", "Кол. 3:14"),
                    ("Брак у всех да будет честен и ложе непорочно.", "Евр. 13:4"),
                ],
                "bible": [
                    ("И будут два одна плоть; так что они уже не двое, но одна плоть.", "Мф. 19:5-6"),
                    ("Так положил Иисус начало чудесам в Кане Галилейской.", "Ин. 2:11"),
                    ("Что Бог сочетал, того человек да не разлучает.", "Мф. 19:6"),
                ],
                "father": [
                    ("Супружество есть таинство любви, где муж и жена становятся соработниками Божией благодати.", "Свт. Иоанн Златоуст"),
                    ("Дом, где муж и жена живут в молитве и мире, становится малой церковью.", "Свт. Феофан Затворник"),
                    ("Верность в браке хранится не силой характера только, но благодатью и покаянным трудом.", "Свт. Тихон Задонский"),
                ],
                "preacher": [
                    ("Христианская семья крепнет там, где супруги вместе молятся, прощают и служат друг другу.", "Свт. Лука (Войно-Ясенецкий)"),
                    ("Счастье супругов рождается там, где каждый учится жертвовать собой ради другого.", "Митрополит Антоний Сурожский"),
                    ("Брак живет тогда, когда муж и жена не соревнуются в правоте, а ищут вместе волю Божию.", "Протоиерей Александр Мень"),
                ],
            }

        if self._is_resurrection_topic(req):
            return {
                "old_testament": [
                    ("Поглощена будет смерть навеки, и отрет Господь Бог слезы со всех лиц.", "Ис. 25:8"),
                    ("Оживит нас через два дня, в третий день восставит нас, и мы будем жить пред лицем Его.", "Ос. 6:2"),
                    ("Это день, который сотворил Господь: возрадуемся и возвеселимся в оный.", "Пс. 117:24"),
                ],
                "apostle": [
                    ("Христос воскрес из мертвых, первенец из умерших.", "1 Кор. 15:20"),
                    ("Пасха наша, Христос, заклан за нас.", "1 Кор. 5:7"),
                    ("Если мы с Ним умерли, то с Ним и оживем.", "2 Тим. 2:11"),
                ],
            }

        quote_bank = self._quote_bank(req)
        return {
            "old_testament": [
                ("Надейся на Господа всем сердцем твоим и не полагайся на разум твой.", "Притч. 3:5"),
                ("О, человек! сказано тебе, что - добро и чего требует от тебя Господь.", "Мих. 6:8"),
                ("Близок Господь ко всем призывающим Его, ко всем призывающим Его в истине.", "Пс. 144:18"),
            ],
            "apostle": [
                ("Всегда радуйтесь. Непрестанно молитесь. За все благодарите.", "1 Фес. 5:16-18"),
                ("Плод же духа: любовь, радость, мир, долготерпение, благость, милосердие, вера.", "Гал. 5:22"),
                ("В скорби будьте терпеливы, в молитве постоянны.", "Рим. 12:12"),
            ],
            "bible": [quote_bank["bible"]],
            "father": [quote_bank["father"]],
            "preacher": [quote_bank["preacher"]],
        }

    def _pick_quote_variant(
        self, options: List[Tuple[str, str]], req: GenerateRequest, kind: str
    ) -> Tuple[str, str]:
        if not options:
            return "", ""
        topic = self._extract_topic(req)
        variant = (req.variant_tag or "").strip().upper()
        variant_offset = {"A": 0, "B": 1, "C": 2}.get(variant)

        # Для A/B/C гарантируем различие при наличии >=2 вариантов.
        if variant_offset is not None and len(options) >= 2:
            topic_seed = int(hashlib.sha256(f"{topic}|{kind}".encode("utf-8")).hexdigest()[:8], 16)
            idx = (topic_seed + variant_offset) % len(options)
            return options[idx]
        # В одиночном режиме добавляем вариативность между запусками,
        # но избегаем мгновенного повтора той же самой цитаты.
        key = f"quote|{topic.lower()}|{kind}"
        idx = self._pick_nonrepeating_index(key, len(options))
        return options[idx]

    def _pick_quote_variants(
        self,
        options: List[Tuple[str, str]],
        req: GenerateRequest,
        kind: str,
        count: int = 1,
    ) -> List[Tuple[str, str]]:
        if not options:
            return []
        if count <= 1:
            return [self._pick_quote_variant(options, req, kind)]

        size = len(options)
        count = max(1, min(count, size))
        topic = self._extract_topic(req)
        variant = (req.variant_tag or "").strip().upper()

        if variant in {"A", "B", "C"} and size >= 2:
            variant_offset = {"A": 0, "B": 1, "C": 2}.get(variant, 0)
            topic_seed = int(hashlib.sha256(f"{topic}|{kind}".encode("utf-8")).hexdigest()[:8], 16)
            start = (topic_seed + variant_offset) % size
            order = options[start:] + options[:start]
            return order[:count]

        key = f"quote_pack|{topic.lower()}|{kind}"
        start = self._pick_nonrepeating_index(key, size)
        order = options[start:] + options[:start]
        return order[:count]

    def _quote_bank(self, req: GenerateRequest) -> Dict[str, Tuple[str, str]]:
        topic_low = self._extract_topic(req).lower()
        feast_sub = self._feast_subtopic(topic_low)
        sin_profile = self._sin_profile(topic_low)
        if self._is_lazarus_topic(req):
            return {
                "bible": (
                    "Я есмь воскресение и жизнь; верующий в Меня, если и умрет, оживет.",
                    "Ин. 11:25",
                ),
                "father": (
                    "Не отчаивайся после падений, но снова и снова начинай покаяние.",
                    "Прп. Иоанн Лествичник",
                ),
                "preacher": (
                    "Когда сердце человека оживает к молитве, он начинает видеть действие Божией благодати во всем.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        if self._is_prodigal_topic(req):
            return {
                "bible": (
                    "Встану, пойду к отцу моему и скажу ему: отче! я согрешил против неба и пред тобою.",
                    "Лк. 15:18",
                ),
                "father": (
                    "Покаяние не унижает человека, но возвращает ему сыновнее достоинство перед Богом.",
                    "Свт. Иоанн Златоуст",
                ),
                "preacher": (
                    "Бог всегда ждет человека с распростертыми объятиями, но путь домой начинается с решимости встать и идти.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        event_profile = self._event_profile(topic_low)
        if event_profile:
            return {
                "bible": tuple(event_profile.get("bible", ("Без Меня не можете делать ничего.", "Ин. 15:5"))),  # type: ignore[arg-type]
                "father": tuple(
                    event_profile.get(
                        "father",
                        ("Ничто так не угодно Богу, как жизнь, посвященная любви и пользе ближнего.", "Свт. Иоанн Златоуст"),
                    )
                ),  # type: ignore[arg-type]
                "preacher": tuple(
                    event_profile.get(
                        "preacher",
                        ("Христианство начинается там, где вера становится ежедневной ответственностью сердца.", "Протоиерей Александр Мень"),
                    )
                ),  # type: ignore[arg-type]
            }
        if any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            return {
                "bible": ("Се, Раба Господня; да будет Мне по слову твоему.", "Лк. 1:38"),
                "father": (
                    "Через Пресвятую Деву к нам пришла радость спасения, и вся Церковь учится у Нее смирению.",
                    "Свт. Григорий Палама",
                ),
                "preacher": (
                    "Почитание Божией Матери всегда приводит человека ко Христу и к более глубокой молитве.",
                    "Протопресвитер Александр Шмеман",
                ),
            }
        if self._is_resurrection_topic(req):
            return {
                "bible": ("Что ищете живого между мертвыми? Его нет здесь: Он воскрес.", "Лк. 24:5-6"),
                "father": (
                    "Христос воскрес, и жизнь торжествует; Христос воскрес, и ни один мертвый не во гробе.",
                    "Свт. Иоанн Златоуст",
                ),
                "preacher": (
                    "Пасхальная радость подлинна тогда, когда она преображает наши отношения с ближними.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        if feast_sub == "trinity":
            return {
                "bible": (
                    "И исполнились все Духа Святаго и начали говорить на иных языках.",
                    "Деян. 2:4",
                ),
                "father": (
                    "Дух Святой не разделяет, но собирает Церковь в любви и истине.",
                    "Свт. Василий Великий",
                ),
                "preacher": (
                    "Пятидесятница учит нас жить так, чтобы сердце было открыто действию благодати каждый день.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        if feast_sub == "entry_jerusalem":
            return {
                "bible": (
                    "Осанна Сыну Давидову! Благословен Грядущий во имя Господне!",
                    "Мф. 21:9",
                ),
                "father": (
                    "Вход Господень в Иерусалим напоминает, что Христос приходит к смиренным сердцам.",
                    "Свт. Феофан Затворник",
                ),
                "preacher": (
                    "Вербное воскресенье призывает нас встречать Христа не ветвями только, но покаянием и верностью.",
                    "Протоиерей Александр Мень",
                ),
            }
        if sin_profile:
            return {
                "bible": tuple(sin_profile.get("apostle", ("Покайтесь и обратитесь, чтобы загладились грехи ваши.", "Деян. 3:19"))),  # type: ignore[arg-type]
                "father": tuple(
                    sin_profile.get(
                        "father",
                        ("Не столько страшно пасть, сколько, пав, оставаться без покаяния.", "Свт. Иоанн Златоуст"),
                    )
                ),  # type: ignore[arg-type]
                "preacher": tuple(
                    sin_profile.get(
                        "preacher",
                        ("Покаяние начинается там, где человек перестает оправдывать грех и берет ответственность перед Богом.", "Свт. Феофан Затворник"),
                    )
                ),  # type: ignore[arg-type]
            }
        if any(w in topic_low for w in ["любов", "милосерд", "прощ", "ближн"]):
            return {
                "bible": (
                    "По тому узнают все, что вы Мои ученики, если будете иметь любовь между собою.",
                    "Ин. 13:35",
                ),
                "father": (
                    "Сердце милующее горит любовью о всей твари, и в этом открывается образ Христов в человеке.",
                    "Прп. Исаак Сирин",
                ),
                "preacher": (
                    "Если слово проповеди пронзило твое сердце, оно коснется и сердца другого.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        if self._is_feast_topic(topic_low):
            return {
                "bible": (
                    "Иисус Христос вчера и сегодня и во веки Тот же.",
                    "Евр. 13:8",
                ),
                "father": (
                    "Церковный праздник дан нам не для внешней радости только, но для обновления сердца и жизни по Евангелию.",
                    "Свт. Феофан Затворник",
                ),
                "preacher": (
                    "Праздник становится подлинным тогда, когда через него мы глубже встречаемся со Христом в молитве и любви.",
                    "Митрополит Антоний Сурожский",
                ),
            }
        if self._is_saint_topic(topic_low):
            return {
                "bible": (
                    "Подражайте мне, как я Христу.",
                    "1 Кор. 4:16",
                ),
                "father": (
                    "Память святых призывает не к восхищению издали, а к подражанию их вере и терпению.",
                    "Свт. Игнатий (Брянчанинов)",
                ),
                "preacher": (
                    "Святые не отрывают нас от повседневности, они учат хранить верность Богу именно в ней.",
                    "Свт. Лука (Войно-Ясенецкий)",
                ),
            }
        return {
            "bible": ("Без Меня не можете делать ничего.", "Ин. 15:5"),
            "father": (
                "Ничто так не угодно Богу, как жизнь, посвященная любви и пользе ближнего.",
                "Свт. Иоанн Златоуст",
            ),
            "preacher": (
                "Христианство начинается там, где вера становится ежедневной ответственностью сердца.",
                "Протоиерей Александр Мень",
            ),
        }

    def _to_genitive_name(self, token: str) -> str:
        word = self.preprocessor.normalize(token)
        low = word.lower()
        irregular = {
            "иоанн": "Иоанна",
            "павел": "Павла",
            "петр": "Петра",
            "пётр": "Петра",
            "лев": "Льва",
            "тихон": "Тихона",
            "феофан": "Феофана",
            "максим": "Максима",
            "игнатий": "Игнатия",
            "василий": "Василия",
            "григорий": "Григория",
            "николай": "Николая",
            "дмитрий": "Димитрия",
            "александр": "Александра",
        }
        if low in irregular:
            return irregular[low]
        if low.endswith("ий"):
            return word[:-2] + "ия"
        if low.endswith("ей"):
            return word[:-2] + "ея"
        if low.endswith("й"):
            return word[:-1] + "я"
        if low.endswith("ь"):
            return word[:-1] + "я"
        if low.endswith("а"):
            return word[:-1] + "ы"
        return word + "а"

    def _to_genitive_adjective(self, token: str) -> str:
        word = self.preprocessor.normalize(token)
        low = word.lower()
        if low.endswith("ский"):
            return word[:-4] + "ского"
        if low.endswith("цкий"):
            return word[:-4] + "цкого"
        if low.endswith("ой"):
            return word[:-2] + "ого"
        if low.endswith("ый"):
            return word[:-2] + "ого"
        if low.endswith("ий"):
            return word[:-2] + "его"
        return word

    def _to_genitive_surname(self, token: str) -> str:
        word = self.preprocessor.normalize(token)
        low = word.lower()
        if low.endswith(("ский", "цкий", "ой", "ый", "ий")):
            return self._to_genitive_adjective(word)
        if low.endswith("ец"):
            return word[:-2] + "ца"
        if low.endswith("ник"):
            return word + "а"
        if low.endswith("ок"):
            return word[:-2] + "ка"
        if low.endswith("й"):
            return word[:-1] + "я"
        if low.endswith("ь"):
            return word[:-1] + "я"
        if low.endswith("а"):
            return word[:-1] + "ы"
        return word + "а"

    def _normalize_author_spelling(self, attribution: str) -> str:
        attr = self.preprocessor.normalize(attribution or "")
        if not attr:
            return attr
        fixes = {
            "Заторник": "Затворник",
            "Феофан Заторник": "Феофан Затворник",
            "Свт. Феофан Заторник": "Свт. Феофан Затворник",
        }
        for wrong, right in fixes.items():
            attr = re.sub(rf"\b{re.escape(wrong)}\b", right, attr)
        return attr

    def _extract_author_prefix(self, attr: str) -> Tuple[str, str]:
        prefixes = ["Свт.", "Прп.", "Блж.", "Свящ.", "Прот.", "Протоиерей", "Митрополит", "Патриарх"]
        for prefix in prefixes:
            if attr.startswith(prefix + " "):
                return prefix, attr[len(prefix) + 1 :].strip()
        return "", attr

    def _inline_author_attribution(self, attribution: str) -> str:
        attr = self._normalize_author_attribution(attribution)
        if not attr:
            return attr
        lower_prefix_map = {
            "Свт. ": "свт. ",
            "Прп. ": "прп. ",
            "Блж. ": "блж. ",
            "Свящ. ": "свящ. ",
            "Прот. ": "прот. ",
            "Протоиерей ": "протоиерей ",
            "Митрополит ": "митрополит ",
            "Патриарх ": "патриарх ",
        }
        for src, dst in lower_prefix_map.items():
            if attr.startswith(src):
                return dst + attr[len(src) :]
        return attr

    def _normalize_author_attribution(self, attribution: str) -> str:
        attr = self._normalize_author_spelling(attribution)
        if not attr:
            return attr
        if any(ch.isdigit() for ch in attr):
            return attr

        prefix, core = self._extract_author_prefix(attr)
        exact = {
            "Кронштадтский Иоанн": "Иоанна Кронштадтского",
            "Златоуст Иоанн": "Иоанна Златоуста",
            "Иоанн Кронштадтский": "Иоанна Кронштадтского",
            "Феофан Затворник": "Феофана Затворника",
        }
        if core in exact:
            normalized = exact[core]
            return f"{prefix} {normalized}".strip() if prefix else normalized

        parts = core.split()
        if len(parts) == 2 and re.fullmatch(r"[А-ЯЁ][а-яё-]+", parts[0]) and re.fullmatch(r"[А-ЯЁ][а-яё-]+", parts[1]):
            left, right = parts
            left_low = left.lower()
            right_low = right.lower()
            if left_low.endswith(("ский", "цкий", "ой", "ый", "ий")):
                gen = f"{self._to_genitive_name(right)} {self._to_genitive_adjective(left)}"
            elif right_low.endswith(("ский", "цкий", "ой", "ый", "ий")):
                gen = f"{self._to_genitive_name(left)} {self._to_genitive_adjective(right)}"
            else:
                gen = f"{self._to_genitive_name(left)} {self._to_genitive_surname(right)}"
            return f"{prefix} {gen}".strip() if prefix else gen
        return f"{prefix} {core}".strip() if prefix else core

    def _format_quote_paragraph(self, kind: str, quote: str, attribution: str) -> str:
        q = self.preprocessor.normalize(quote).strip(" .")
        if q and q[-1] not in ".!?":
            q = q + "."
        attr = self._normalize_author_attribution(attribution)
        if kind == "bible":
            if attr:
                return f"Священное Писание говорит: «{q}» ({attr})."
            return f"Священное Писание говорит: «{q}»."
        if kind == "father":
            if attr:
                return f"{attr} наставляет: «{q}»."
            return f"Святые отцы учат: «{q}»."
        if attr:
            return f"В проповеднической традиции звучит слово {attr}: «{q}»."
        return f"В проповеднической традиции звучит слово: «{q}»."

    def _build_quote_paragraphs(self, req: GenerateRequest, citations: List[Citation]) -> List[str]:
        topic_raw = self._extract_topic(req)
        markers = self._topic_markers(topic_raw)
        specific_keywords = self._topic_specific_keywords(topic_raw)
        event_profile = self._event_profile(topic_raw.lower())
        sin_profile = self._sin_profile(topic_raw.lower())
        prefer_fallback_quotes = event_profile is not None or sin_profile is not None
        topic_words = [
            w
            for w in re.findall(r"[а-яёa-z]{4,}", topic_raw.lower())
            if w not in {"проповедь", "тема", "теме", "тему", "о", "про", "на", "и", "для", "об", "по"}
        ]
        quote_markers = list(dict.fromkeys(topic_words[:6] + markers))
        if specific_keywords:
            quote_markers = list(dict.fromkeys(specific_keywords + quote_markers))
        grouped: Dict[str, List[Citation]] = {"bible": [], "father": [], "preacher": []}
        for c in citations:
            kind = self._source_kind(c.source_type)
            if kind in grouped:
                grouped[kind].append(c)

        quote_bank_alts = self._quote_bank_alternatives(req)

        if sin_profile:
            old_testament_pairs = self._pick_quote_variants(
                quote_bank_alts.get(
                    "old_testament",
                    [
                        (
                            "Скрывающий свои преступления не будет иметь успеха, а кто сознается и оставляет их, тот будет помилован.",
                            "Притч. 28:13",
                        )
                    ],
                ),
                req,
                "old_testament",
                count=2,
            )
            apostle_pairs = self._pick_quote_variants(
                quote_bank_alts.get(
                    "apostle",
                    [("Отложите прежний образ жизни ветхого человека и обновитесь духом ума вашего.", "Еф. 4:22-23")],
                ),
                req,
                "apostle",
                count=2,
            )
            father_quote, father_attr = self._pick_quote_variant(
                quote_bank_alts.get(
                    "father",
                    [("Не столько страшно пасть, сколько, пав, оставаться без покаяния.", "Свт. Иоанн Златоуст")],
                ),
                req,
                "father",
            )
            preacher_quote, preacher_attr = self._pick_quote_variant(
                quote_bank_alts.get(
                    "preacher",
                    [("Покаяние начинается там, где человек перестает оправдывать грех.", "Свт. Феофан Затворник")],
                ),
                req,
                "preacher",
            )
            old_testament_quote = " ".join(self.preprocessor.normalize(q) for q, _ in old_testament_pairs if q).strip()
            old_testament_ref = "; ".join(self.preprocessor.normalize(r) for _, r in old_testament_pairs if r).strip()
            apostle_quote = " ".join(self.preprocessor.normalize(q) for q, _ in apostle_pairs if q).strip()
            apostle_ref = "; ".join(self.preprocessor.normalize(r) for _, r in apostle_pairs if r).strip()

            out = [
                f"Ветхий Завет предупреждает: «{self.preprocessor.normalize(old_testament_quote)}» ({self.preprocessor.normalize(old_testament_ref)}).",
                f"Послание святых апостолов наставляет: «{self.preprocessor.normalize(apostle_quote)}» ({self.preprocessor.normalize(apostle_ref)}).",
                self._format_quote_paragraph("father", father_quote, father_attr),
                self._format_quote_paragraph("preacher", preacher_quote, preacher_attr),
            ]
            return self._dedupe_paragraphs([self._apply_orthodox_casing(p) for p in out if p.strip()])

        quote_bank = self._quote_bank(req)
        out: List[str] = []

        old_testament_pairs = self._pick_quote_variants(
            quote_bank_alts.get(
                "old_testament",
                [("Надейся на Господа всем сердцем твоим и не полагайся на разум твой.", "Притч. 3:5")],
            ),
            req,
            "old_testament",
            count=2,
        )
        apostle_pairs = self._pick_quote_variants(
            quote_bank_alts.get(
                "apostle",
                [("Всегда радуйтесь. Непрестанно молитесь. За все благодарите.", "1 Фес. 5:16-18")],
            ),
            req,
            "apostle",
            count=2,
        )
        old_testament_quote = " ".join(self.preprocessor.normalize(q) for q, _ in old_testament_pairs if q).strip()
        old_testament_ref = "; ".join(self.preprocessor.normalize(r) for _, r in old_testament_pairs if r).strip()
        apostle_quote = " ".join(self.preprocessor.normalize(q) for q, _ in apostle_pairs if q).strip()
        apostle_ref = "; ".join(self.preprocessor.normalize(r) for _, r in apostle_pairs if r).strip()
        out.append(
            f"Ветхий Завет свидетельствует: «{self.preprocessor.normalize(old_testament_quote)}» ({self.preprocessor.normalize(old_testament_ref)})."
        )
        out.append(
            f"Послание святых апостолов наставляет: «{self.preprocessor.normalize(apostle_quote)}» ({self.preprocessor.normalize(apostle_ref)})."
        )

        for kind in ["bible", "father", "preacher"]:
            chosen_quote: Optional[str] = None
            chosen_attr: Optional[str] = None
            variant = (req.variant_tag or "").strip().upper()
            force_fallback_for_variant = (
                kind == "bible" and variant in {"A", "B", "C"} and len(quote_bank_alts.get("bible", [])) >= 2
            )
            if grouped[kind] and not prefer_fallback_quotes and not force_fallback_for_variant:
                ordered = sorted(
                    grouped[kind],
                    key=lambda c: self._citation_relevance_score(c, markers),
                    reverse=True,
                )
                if variant in {"A", "B", "C"}:
                    variant_offset = {"A": 0, "B": 1, "C": 2}.get(variant, 0)
                    rotated = ordered[variant_offset:] + ordered[:variant_offset]
                else:
                    head_len = min(4, len(ordered))
                    head = ordered[:head_len]
                    tail = ordered[head_len:]
                    if head:
                        key = f"cit_pick|{self._extract_topic(req).lower()}|{kind}"
                        start = self._pick_nonrepeating_index(key, len(head))
                        head = head[start:] + head[:start]
                    rotated = head + tail
                for item in rotated:
                    quote = self._extract_quote_sentence(
                        item.excerpt,
                        quote_markers,
                        require_marker=True,
                        specific_markers=specific_keywords if specific_keywords else None,
                    )
                    if quote:
                        chosen_quote = quote
                        if kind == "bible":
                            chosen_attr = item.reference or item.title or "Священное Писание"
                        else:
                            chosen_attr = item.author or item.title or item.reference or ""
                        break

            if not chosen_quote:
                if kind in quote_bank_alts and quote_bank_alts[kind]:
                    fallback_quote, fallback_attr = self._pick_quote_variant(quote_bank_alts[kind], req, kind)
                else:
                    fallback_quote, fallback_attr = quote_bank[kind]
                chosen_quote = fallback_quote
                chosen_attr = fallback_attr

            out.append(self._format_quote_paragraph(kind, chosen_quote, chosen_attr or ""))

        return self._dedupe_paragraphs([self._apply_orthodox_casing(p) for p in out if p.strip()])

    def _ensure_quote_paragraphs(self, sermon: str, req: GenerateRequest, citations: List[Citation]) -> str:
        intro, main, concl = self._split_sermon_sections(sermon)
        if not (intro and main and concl):
            return sermon

        main_low = main.lower()
        has_old_testament = "ветхий завет" in main_low
        has_scripture = "священное писание говорит" in main_low
        has_apostle = "послание святых апостолов наставляет" in main_low
        has_father = ("наставляет: «" in main_low and ("свт." in main_low or "свят" in main_low or "прп." in main_low))
        has_preacher = (
            "проповеднической традиции звучит слово" in main_low
            or "митрополит" in main_low
            or "протоиерей" in main_low
            or "священник" in main_low
        )
        if has_old_testament and has_scripture and has_apostle and has_father and has_preacher:
            return sermon

        add_parts = self._build_quote_paragraphs(req, citations)
        if not add_parts:
            return sermon

        main_parts = [p.strip() for p in main.split("\n\n") if p.strip()]
        existing_keys = {self._sentence_key(p) for p in main_parts}
        for p in add_parts:
            key = self._sentence_key(p)
            if key in existing_keys:
                continue
            main_parts.append(p)
            existing_keys.add(key)

        main = "\n\n".join(main_parts)
        title = self._compose_title(req)
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{concl}"
        )

    def _sentence_key(self, sentence: str) -> str:
        key = self.preprocessor.normalize(sentence).lower()
        key = re.sub(r"[^а-яёa-z0-9 ]+", "", key)
        key = re.sub(r"\s+", " ", key).strip()
        return key

    def _dedupe_sentences(self, text: str) -> str:
        sents = self.preprocessor.split_into_sentences(text)
        if not sents:
            return self.preprocessor.normalize(text)
        seen = set()
        out: List[str] = []
        for s in sents:
            key = self._sentence_key(s)
            if len(key) < 12:
                out.append(s)
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return " ".join(out).strip()

    def _dedupe_paragraphs(self, paragraphs: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for p in paragraphs:
            key = self._sentence_key(p)[:180]
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    def _is_quote_paragraph(self, paragraph: str) -> bool:
        low = self.preprocessor.normalize(paragraph or "").lower()
        markers = [
            "ветхий завет",
            "послание святых апостолов",
            "священное писание говорит",
            "в проповеднической традиции звучит слово",
            "наставляет: «",
        ]
        return any(m in low for m in markers)

    def _rebuild_sermon(self, req: GenerateRequest, intro: str, main: str, concl: str) -> str:
        title = self._compose_title(req)
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro.strip()}\n\n"
            f"Основная часть.\n{main.strip()}\n\n"
            f"Заключение.\n{concl.strip()}"
        )

    def _topic_lock_is_strict(self, req: GenerateRequest) -> bool:
        topic_low = self._extract_topic(req).lower()
        return bool(
            self._sin_profile(topic_low)
            or self._event_profile(topic_low)
            or self._is_lazarus_topic(req)
            or self._is_prodigal_topic(req)
            or self._is_marriage_topic_low(topic_low)
            or self._is_resurrection_topic(req)
            or any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"])
        )

    def _rotate_topic_lock_paragraphs(self, key: str, options: List[str], count: int = 2) -> List[str]:
        if not options:
            return []
        count = max(1, min(count, len(options)))
        if len(options) <= count:
            return options
        start = self._pick_nonrepeating_index(f"topic_lock|{key}", len(options))
        rotated = options[start:] + options[:start]
        return rotated[:count]

    def _topic_lock_extra_paragraphs(self, req: GenerateRequest) -> List[str]:
        topic = self._extract_topic(req)
        topic_low = topic.lower()
        event = self._event_profile(topic_low)
        sin_profile = self._sin_profile(topic_low)
        if sin_profile:
            sin_name = str(sin_profile.get("name_genitive", "греха"))
            focus = str(sin_profile.get("focus", "грех разрушает душу"))
            practice = str(
                sin_profile.get(
                    "practice",
                    "честная исповедь, отказ от поводов ко греху и постоянная молитва",
                )
            )
            options = [
                f"Сегодня Церковь призывает говорить именно о {sin_name}: {focus}.",
                f"Путь исправления здесь конкретен: {practice}; без этого тема останется только словами.",
                f"Говоря о {sin_name}, важно назвать духовные последствия прямо: {focus}, если не последует покаянного труда.",
                f"Практический ответ на тему {sin_name} должен быть ясным и деятельным: {practice}.",
            ]
            return self._rotate_topic_lock_paragraphs(f"sin|{topic_low}", options, count=2)
        if event:
            name = str(event.get("name", topic))
            focus = str(event.get("focus", "тема требует личного ответа веры"))
            practice = str(event.get("practice", "жить по Евангелию в повседневности"))
            options = [
                f"Тема «{name}» требует не общего рассуждения, а личного духовного ответа: {focus}.",
                f"Именно в контексте «{name}» становится ясно, что практический путь для нас сегодня таков: {practice}.",
                f"Праздничная тема «{name}» раскрывается полно только тогда, когда мы принимаем ее в жизнь: {focus}.",
                f"Для нашей общины главное в теме «{name}» — не теория, а конкретная верность: {practice}.",
            ]
            return self._rotate_topic_lock_paragraphs(f"event|{topic_low}", options, count=2)
        if self._is_lazarus_topic(req):
            options = [
                "Лазарева суббота раскрывает евангельское свидетельство о Лазаре Четверодневном и о власти Христа над смертью.",
                "Через событие в Вифании Церковь укрепляет нас надеждой на воскресение и призывает к деятельному покаянию.",
                "В центре Лазаревой субботы стоит призыв Христа к жизни, обращенный и к нашей совести, когда она омертвевает от греха.",
                "Событие в Вифании напоминает: Господь возвращает жизнь там, где человек с верой откликается на Божий зов.",
            ]
            return self._rotate_topic_lock_paragraphs(f"lazar|{topic_low}", options, count=2)
        if self._is_prodigal_topic(req):
            options = [
                "Притча о блудном сыне говорит не о постороннем герое, а о каждом из нас, когда мы удаляемся от Отца Небесного.",
                "Возвращение блудного сына к отцу показывает путь покаяния: признание греха, перемена жизни и восстановление любви.",
                "Блудный сын учит нас, что путь домой начинается с честного признания собственной вины перед Богом.",
                "Отец из притчи о блудном сыне открывает нам образ милости Божией, принимающей кающегося без отвержения.",
            ]
            return self._rotate_topic_lock_paragraphs(f"prodigal|{topic_low}", options, count=2)
        if self._is_marriage_topic_low(topic_low):
            options = [
                "В теме Таинства Венчания важно прямо сказать о взаимной ответственности супругов перед Богом и Церковью.",
                "Христианский брак как таинство раскрывается в верности, совместной молитве и жертвенной любви мужа и жены.",
                "Говоря о Венчании, необходимо подчеркнуть, что супруги призваны не только к союзу чувств, но и к совместному пути спасения.",
                "Таинство брака требует ежедневной верности: терпения, прощения и общей молитвенной жизни семьи перед Богом.",
            ]
            return self._rotate_topic_lock_paragraphs(f"marriage|{topic_low}", options, count=2)
        if any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            options = [
                "Почитание Пресвятой Богородицы всегда ведет ко Христу и учит смирению, чистоте сердца и молитвенному доверию Богу.",
                "Образ Божией Матери раскрывает для нас материнское заступничество и призыв к внутреннему трезвению.",
                "Богородичная тема требует личного ответа веры: хранить мир в сердце и чаще обращаться к Матери Божией в молитве.",
                "Через почитание Божией Матери Церковь воспитывает в нас благодарение, кротость и верность Христу.",
            ]
            return self._rotate_topic_lock_paragraphs(f"theotokos|{topic_low}", options, count=2)
        return []

    def _enforce_topic_lock(self, sermon: str, req: GenerateRequest) -> str:
        intro, main, concl = self._split_sermon_sections(sermon)
        if not (intro and main and concl):
            return sermon
        markers = self._topic_markers(self._extract_topic(req))
        if not markers:
            return sermon

        main_low = self.preprocessor.normalize(main).lower()
        unique_markers = [m for m in markers if m]
        hits = sum(1 for m in unique_markers if m in main_low)
        strict = self._topic_lock_is_strict(req)
        min_hits = min(len(unique_markers), 3 if strict else 2)
        if hits >= min_hits:
            return sermon
        if hits >= max(1, min_hits - 1) and self._topic_is_covered(sermon, req):
            return sermon

        main_parts = [p.strip() for p in main.split("\n\n") if p.strip()]
        existing = {self._sentence_key(p) for p in main_parts}
        for paragraph in self._topic_lock_extra_paragraphs(req):
            paragraph_norm = self.preprocessor.normalize(paragraph)
            if not paragraph_norm:
                continue
            key = self._sentence_key(paragraph_norm)
            if key in existing:
                continue
            main_parts.append(paragraph_norm)
            existing.add(key)

        main_parts = self._dedupe_paragraphs(main_parts)
        main = "\n\n".join(main_parts).strip()
        return self._rebuild_sermon(req, intro, main, concl)

    def _tighten_main_repetition(self, sermon: str, req: GenerateRequest) -> str:
        intro, main, concl = self._split_sermon_sections(sermon)
        if not (intro and main and concl):
            return sermon
        main_low = self.preprocessor.normalize(main).lower()
        cliche_hits = sum(main_low.count(m) for m in CLICHE_MARKERS)
        if self._repetition_penalty(sermon) < 3.0 and cliche_hits <= 1:
            return sermon
        paragraphs = [p.strip() for p in main.split("\n\n") if p.strip()]
        if not paragraphs:
            return sermon

        sentence_seen = set()
        paragraph_seen = set()
        reference_seen = set()
        cliche_seen = set()
        cleaned: List[str] = []

        for paragraph in paragraphs:
            text = self.preprocessor.normalize(paragraph)
            if not text:
                continue
            text_low = text.lower()

            # Убираем повторяющиеся клишированные маркеры в пределах одной проповеди.
            skip_paragraph = False
            for marker in CLICHE_MARKERS:
                if marker in text_low:
                    if marker in cliche_seen:
                        skip_paragraph = True
                        break
                    cliche_seen.add(marker)
            if skip_paragraph:
                continue

            # Для блоков с цитатами избегаем повтора одной и той же библейской ссылки.
            if self._is_quote_paragraph(text):
                refs = re.findall(r"\(([^\)]{3,64})\)", text)
                if refs:
                    ref_key = self._sentence_key(";".join(refs))
                    if ref_key in reference_seen:
                        continue
                    reference_seen.add(ref_key)

            local_sentences = []
            for sentence in self.preprocessor.split_into_sentences(text):
                sent_key = self._sentence_key(sentence)
                if len(sent_key) < 12:
                    continue
                if sent_key in sentence_seen:
                    continue
                sentence_seen.add(sent_key)
                local_sentences.append(sentence.strip())
            if not local_sentences:
                continue

            merged = " ".join(local_sentences).strip()
            if not merged:
                continue
            par_key = self._sentence_key(merged)[:220]
            if par_key in paragraph_seen:
                continue
            paragraph_seen.add(par_key)

            cleaned.append(merged)

        if len(cleaned) < max(6, len(paragraphs) // 2):
            cleaned = self._dedupe_paragraphs([self.preprocessor.normalize(p) for p in paragraphs])

        main = "\n\n".join(cleaned).strip()
        return self._rebuild_sermon(req=req, intro=intro, main=main, concl=concl)

    def _connective_prefix(self, idx: int, is_sin_topic: bool) -> str:
        common = [
            "Далее",
            "При этом",
            "Именно поэтому",
            "Кроме того",
            "Отсюда следует",
            "Вместе с тем",
            "Наконец",
        ]
        strict = [
            "Здесь особенно важно увидеть:",
            "Поэтому недопустимо самооправдание:",
            "Именно в этом месте совесть должна отрезвиться:",
            "Следовательно, требуется конкретное исправление:",
            "Итак, практический вывод предельно ясен:",
        ]
        pool = strict if is_sin_topic else common
        return pool[idx % len(pool)]

    def _add_cohesive_transitions(self, paragraphs: List[str], is_sin_topic: bool) -> List[str]:
        out: List[str] = []
        for i, p in enumerate(paragraphs):
            text = self.preprocessor.normalize(p or "")
            if not text:
                continue
            if i == 0 or self._is_quote_paragraph(text):
                out.append(text)
                continue
            if re.match(
                r"^(Далее|При этом|Именно поэтому|Кроме того|Отсюда следует|Вместе с тем|Наконец|Итак|Следовательно|Здесь особенно важно увидеть)\b",
                text,
                flags=re.IGNORECASE,
            ):
                out.append(text)
                continue
            prefix = self._connective_prefix(i, is_sin_topic)
            if text[0].islower():
                text = text[0].upper() + text[1:]
            if prefix.endswith(":"):
                out.append(f"{prefix} {text}")
            else:
                out.append(f"{prefix}, {text[0].lower() + text[1:]}")
        return out

    def _main_is_substantial(self, sermon: str) -> bool:
        _, main, _ = self._split_sermon_sections(sermon)
        if not main:
            return False
        paragraphs = [p.strip() for p in main.split("\n\n") if p.strip()]
        if len(paragraphs) < 5:
            return False
        words = re.findall(r"[А-Яа-яA-Za-zЁё]+", main)
        if len(words) < 230:
            return False
        return True

    def _sermon_signature(self, sermon: str) -> str:
        _, main, _ = self._split_sermon_sections(sermon or "")
        base = self.preprocessor.normalize(main or sermon or "").lower()
        words = [
            w
            for w in re.findall(r"[а-яёa-z]{4,}", base)
            if w
            not in {
                "вступление",
                "основная",
                "часть",
                "заключение",
                "сегодня",
                "церковь",
                "господь",
                "христос",
                "аминь",
                "дорогие",
                "братья",
                "сестры",
            }
        ]
        return " ".join(words[:120]).strip()

    def _lexical_variety(self, sermon: str) -> float:
        _, main, _ = self._split_sermon_sections(sermon or "")
        base = self.preprocessor.normalize(main or sermon or "").lower()
        words = re.findall(r"[а-яёa-z]{4,}", base)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _repetition_penalty(self, sermon: str) -> float:
        _, main, _ = self._split_sermon_sections(sermon or "")
        text = main or sermon or ""
        sents = self.preprocessor.split_into_sentences(text)
        if not sents:
            return 0.0
        counts: Dict[str, int] = {}
        for s in sents:
            key = self._sentence_key(s)[:180]
            if len(key) < 18:
                continue
            counts[key] = counts.get(key, 0) + 1
        repeats = sum(max(0, c - 1) for c in counts.values())
        return float(repeats)

    def _sermon_quality_score(self, sermon: str, req: GenerateRequest) -> float:
        if not sermon:
            return -1e9
        low = (sermon or "").lower()
        score = 0.0

        if self._is_noisy_sermon(sermon, require_structure_markers=False):
            score -= 90.0
        else:
            score += 26.0

        if self._is_structured_sermon(sermon):
            score += 24.0
        else:
            score -= 28.0

        if self._topic_is_covered(sermon, req):
            score += 42.0
        else:
            score -= 70.0

        if self._main_is_substantial(sermon):
            score += 30.0
        else:
            score -= 24.0

        variety = self._lexical_variety(sermon)
        score += variety * 62.0

        score -= self._repetition_penalty(sermon) * 7.5

        cliche_hits = sum(low.count(m) for m in CLICHE_MARKERS)
        score -= cliche_hits * 8.0

        signature = self._sermon_signature(sermon)
        if signature:
            if signature in self._recent_sermon_signatures:
                score -= 22.0
            else:
                score += 12.0

        max_overlap = 0.0
        for prev in self._recent_sermons[-8:]:
            max_overlap = max(max_overlap, self._sentence_overlap_ratio(sermon, prev))
        if max_overlap >= 0.84:
            score -= 32.0
        elif max_overlap >= 0.72:
            score -= 16.0
        elif max_overlap <= 0.4:
            score += 4.0

        words_count = len(re.findall(r"[А-Яа-яA-Za-zЁё]+", sermon))
        score += min(words_count, 520) / 26.0
        return score

    def _pick_best_candidate(self, candidates: List[str], req: GenerateRequest) -> str:
        uniq: List[str] = []
        seen = set()
        for item in candidates:
            text = (item or "").strip()
            if not text:
                continue
            sig = self._sermon_signature(text)
            key = sig[:260] if sig else self._sentence_key(text)[:260]
            if key in seen:
                continue
            seen.add(key)
            uniq.append(text)
        if not uniq:
            return ""
        return max(uniq, key=lambda s: self._sermon_quality_score(s, req))

    def _remember_sermon(self, sermon: str) -> None:
        sig = self._sermon_signature(sermon)
        if not sig:
            return
        self._recent_sermon_signatures.append(sig)
        self._recent_sermon_signatures = self._recent_sermon_signatures[-60:]
        self._recent_sermons.append((sermon or "").strip())
        self._recent_sermons = self._recent_sermons[-30:]

    def _sentence_key_set(self, text: str) -> set:
        sents = self.preprocessor.split_into_sentences(text or "")
        keys = {
            self._sentence_key(s)[:180]
            for s in sents
            if len(re.findall(r"[А-Яа-яA-Za-zЁё]+", s)) >= 5
        }
        if keys:
            return keys
        words = re.findall(r"[А-Яа-яA-Za-zЁё]{4,}", (text or "").lower())
        if not words:
            return set()
        return {" ".join(words[:80])}

    def _sentence_overlap_ratio(self, left: str, right: str) -> float:
        left_keys = self._sentence_key_set(left)
        right_keys = self._sentence_key_set(right)
        if not left_keys or not right_keys:
            return 0.0
        inter = len(left_keys & right_keys)
        union = len(left_keys | right_keys)
        return float(inter / union) if union else 0.0

    def _is_too_similar_to_recent(self, sermon: str, threshold: float = 0.78) -> bool:
        if not sermon or not self._recent_sermons:
            return False
        for prev in self._recent_sermons[-6:]:
            if self._sentence_overlap_ratio(sermon, prev) >= threshold:
                return True
        return False

    def _pick_rotating_index(self, key: str, size: int) -> int:
        if size <= 1:
            return 0
        state = self._rotation_state.get(key)
        if not state or int(state.get("size", -1)) != size:
            order = list(range(size))
            random.SystemRandom().shuffle(order)
            state = {"size": size, "order": order, "pos": 0, "last": -1}
            self._rotation_state[key] = state

        order = list(state.get("order", list(range(size))))
        pos = int(state.get("pos", 0))
        if pos >= len(order):
            prev = int(state.get("last", -1))
            random.SystemRandom().shuffle(order)
            if len(order) > 1 and order[0] == prev:
                order.append(order.pop(0))
            pos = 0

        idx = int(order[pos])
        state["order"] = order
        state["pos"] = pos + 1
        state["last"] = idx
        self._rotation_state[key] = state
        return idx

    def _pick_nonrepeating_index(self, key: str, size: int) -> int:
        if size <= 1:
            return 0
        idx = self._pick_rotating_index(key, size)
        self._recent_choice_index[key] = idx
        return idx

    def _diversify_citations(self, citations: List[Citation], req: GenerateRequest) -> List[Citation]:
        if len(citations) <= 2:
            return citations
        topic = self._extract_topic(req).lower()
        variant = (req.variant_tag or "").strip().upper()

        if variant in {"A", "B", "C"}:
            offset = {"A": 0, "B": 1, "C": 2}.get(variant, 0)
            offset = offset % len(citations)
            return citations[offset:] + citations[:offset]

        key = f"cit_order|{topic}"
        start = self._pick_nonrepeating_index(key, len(citations))
        return citations[start:] + citations[:start]

    def _select_citation_window(self, citations: List[Citation], req: GenerateRequest) -> List[Citation]:
        target = max(1, min(req.top_k_sources, 10))
        if len(citations) <= target:
            return citations[:target]

        topic = self._extract_topic(req).lower()
        variant = (req.variant_tag or "").strip().upper()
        head = citations[:1]
        pool = citations[1 : max(target * 4, target + 8)]
        if not pool:
            return citations[:target]

        if variant in {"A", "B", "C"}:
            start = {"A": 0, "B": 1, "C": 2}.get(variant, 0) % len(pool)
        else:
            start = self._pick_nonrepeating_index(f"cit_window|{topic}", len(pool))
        rotated = pool[start:] + pool[:start]

        selected: List[Citation] = []
        seen_ids = set()
        seen_meta = set()
        per_type: Dict[str, int] = {}
        max_per_type = max(1, target // 2)

        for item in rotated:
            if len(selected) >= target - 1:
                break
            item_id = (item.id or "").strip()
            type_key = (item.source_type or "unknown").lower()
            meta_key = (
                type_key,
                self.preprocessor.normalize(item.author or item.title or "").lower(),
                self.preprocessor.normalize(item.reference or "").lower(),
            )
            if item_id and item_id in seen_ids:
                continue
            if meta_key in seen_meta:
                continue
            if per_type.get(type_key, 0) >= max_per_type:
                continue
            selected.append(item)
            if item_id:
                seen_ids.add(item_id)
            seen_meta.add(meta_key)
            per_type[type_key] = per_type.get(type_key, 0) + 1

        if len(selected) < target - 1:
            for item in rotated:
                if len(selected) >= target - 1:
                    break
                item_id = (item.id or "").strip()
                type_key = (item.source_type or "unknown").lower()
                meta_key = (
                    type_key,
                    self.preprocessor.normalize(item.author or item.title or "").lower(),
                    self.preprocessor.normalize(item.reference or "").lower(),
                )
                if item_id and item_id in seen_ids:
                    continue
                if meta_key in seen_meta:
                    continue
                selected.append(item)
                if item_id:
                    seen_ids.add(item_id)
                seen_meta.add(meta_key)

        return head + selected

    def _word_count(self, text: str) -> int:
        return len(re.findall(r"[А-Яа-яA-Za-zЁё]+", text or ""))

    def _band_score(self, value: int, low: int, high: int, hard_low: int, hard_high: int) -> float:
        if value <= hard_low or value >= hard_high:
            return 0.0
        if low <= value <= high:
            return 100.0
        if value < low:
            span = max(1, low - hard_low)
            return max(0.0, 100.0 * (value - hard_low) / span)
        span = max(1, hard_high - high)
        return max(0.0, 100.0 * (hard_high - value) / span)

    def _topic_relevance_score(self, sermon: str, req: GenerateRequest) -> float:
        topic = self._extract_topic(req)
        markers = list(dict.fromkeys(self._topic_markers(topic) + self._topic_specific_keywords(topic)))
        _, main, _ = self._split_sermon_sections(sermon)
        text_low = self.preprocessor.normalize(sermon).lower()
        main_low = self.preprocessor.normalize(main or sermon).lower()
        if not markers:
            return 92.0 if self._topic_is_covered(sermon, req) else 58.0

        total = len(markers)
        text_hits = sum(1 for m in markers if m and m in text_low)
        main_hits = sum(1 for m in markers if m and m in main_low)
        weighted = ((text_hits / total) * 0.35) + ((main_hits / total) * 0.65)
        base = 48.0 + (weighted * 52.0)

        if self._topic_is_covered(sermon, req):
            base += 6.0
        else:
            base -= 12.0

        return self._clamp_score(base)

    def _structure_quality_score(self, sermon: str) -> float:
        low = (sermon or "").lower()
        has_title = low.startswith("проповедь на тему:") or low.startswith("проповедь:")
        intro, main, concl = self._split_sermon_sections(sermon)
        if not (intro and main and concl):
            return 28.0

        intro_w = self._word_count(intro)
        main_w = self._word_count(main)
        concl_w = self._word_count(concl)
        intro_s = self._band_score(intro_w, low=42, high=150, hard_low=14, hard_high=260)
        main_s = self._band_score(main_w, low=260, high=760, hard_low=110, hard_high=1200)
        concl_s = self._band_score(concl_w, low=38, high=170, hard_low=12, hard_high=300)

        intro_sent = len(self.preprocessor.split_into_sentences(intro))
        main_sent = len(self.preprocessor.split_into_sentences(main))
        concl_sent = len(self.preprocessor.split_into_sentences(concl))
        sentence_s = self._band_score(intro_sent + main_sent + concl_sent, low=10, high=24, hard_low=4, hard_high=40)

        header_bonus = 6.0 if has_title else 0.0
        strict_bonus = 7.0 if self._is_structured_sermon(sermon) else 0.0
        return self._clamp_score(intro_s * 0.2 + main_s * 0.5 + concl_s * 0.2 + sentence_s * 0.1 + header_bonus + strict_bonus)

    def _substance_quality_score(self, sermon: str) -> float:
        _, main, _ = self._split_sermon_sections(sermon)
        text = main or sermon
        words = self._word_count(text)
        sents = self.preprocessor.split_into_sentences(text)
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        length_s = self._band_score(words, low=280, high=860, hard_low=120, hard_high=1400)
        sent_s = self._band_score(len(sents), low=9, high=26, hard_low=4, hard_high=45)
        para_s = self._band_score(len(paragraphs), low=6, high=18, hard_low=3, hard_high=30)

        low = text.lower()
        quote_markers = [
            "ветхий завет",
            "послание святых апостолов",
            "священное писание говорит",
            "проповеднической традиции звучит слово",
            "наставляет: «",
        ]
        quote_hits = sum(1 for m in quote_markers if m in low)
        quote_s = min(100.0, quote_hits * 22.0)

        return self._clamp_score(length_s * 0.45 + sent_s * 0.2 + para_s * 0.15 + quote_s * 0.2)

    def _diversity_quality_score(self, sermon: str) -> float:
        _, main, _ = self._split_sermon_sections(sermon)
        text = main or sermon
        words = re.findall(r"[а-яёa-z]{4,}", self.preprocessor.normalize(text).lower())
        if not words:
            return 0.0

        lexical = self._lexical_variety(sermon)
        lexical_s = min(100.0, lexical * 145.0)

        bigrams = [" ".join(words[i : i + 2]) for i in range(max(0, len(words) - 1))]
        bigram_ratio = (len(set(bigrams)) / len(bigrams)) if bigrams else 0.0
        bigram_s = min(100.0, bigram_ratio * 120.0)

        sents = self.preprocessor.split_into_sentences(text)
        sent_keys = [self._sentence_key(s)[:180] for s in sents if self._word_count(s) >= 5]
        sent_ratio = (len(set(sent_keys)) / len(sent_keys)) if sent_keys else 1.0
        sentence_s = min(100.0, sent_ratio * 100.0)

        return self._clamp_score(lexical_s * 0.5 + bigram_s * 0.3 + sentence_s * 0.2)

    def _repetition_control_quality_score(self, sermon: str) -> float:
        low = (sermon or "").lower()
        repeat_penalty = self._repetition_penalty(sermon)
        score = 100.0 - repeat_penalty * 13.0

        max_overlap = 0.0
        for prev in self._recent_sermons[-8:]:
            max_overlap = max(max_overlap, self._sentence_overlap_ratio(sermon, prev))
        score -= max_overlap * 48.0

        score -= sum(low.count(m) for m in CLICHE_MARKERS) * 6.0
        return self._clamp_score(score)

    def _clamp_score(self, value: float) -> float:
        return round(max(0.0, min(100.0, value)), 2)

    def _build_quality_metrics(self, sermon: str, req: GenerateRequest) -> QualityMetrics:
        topic_relevance = self._topic_relevance_score(sermon, req)
        structure_score = self._structure_quality_score(sermon)
        substance_score = self._substance_quality_score(sermon)
        diversity_score = self._diversity_quality_score(sermon)
        repetition_control_score = self._repetition_control_quality_score(sermon)

        overall = (
            topic_relevance * 0.28
            + structure_score * 0.2
            + substance_score * 0.24
            + diversity_score * 0.16
            + repetition_control_score * 0.12
        )

        notes: List[str] = []
        if topic_relevance < 72:
            notes.append("Тема раскрыта частично: стоит уточнить формулировку промта.")
        if structure_score < 74:
            notes.append("Структура могла просесть: проверьте полноту вступления, основной части и заключения.")
        if substance_score < 68:
            notes.append("Содержательность умеренная: попросите раскрыть тему глубже и добавить пастырские примеры.")
        if diversity_score < 64:
            notes.append("Текст местами однотипный: попробуйте шаблон «Праздничная» или «Обычная».")
        if repetition_control_score < 72:
            notes.append("Есть повторяющиеся фразы: добавьте больше контекста в промт.")

        return QualityMetrics(
            overall_score=self._clamp_score(overall),
            topic_relevance=self._clamp_score(topic_relevance),
            structure_score=self._clamp_score(structure_score),
            substance_score=self._clamp_score(substance_score),
            diversity_score=self._clamp_score(diversity_score),
            repetition_control_score=self._clamp_score(repetition_control_score),
            notes=notes[:3],
        )

    def _compose_safe_sermon(self, req: GenerateRequest, citations: List[Citation]) -> str:
        topic = self._extract_topic(req)
        topic = self._apply_orthodox_casing(topic)
        bible_ref = self.preprocessor.normalize(req.bible_text or "")
        occasion = self.preprocessor.normalize(req.occasion or "")
        occasion_provided = bool(occasion)
        audience = self.preprocessor.normalize(req.audience or "прихода")
        audience_low = audience.lower()

        seed_src = f"{topic}|{bible_ref}|{occasion}|{audience}|{random.SystemRandom().randint(1, 2**31 - 1)}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)
        # Нерепродуцируемая вариативность между запросами.
        rng = random.Random(seed)

        def pick(options: List[str], salt: int = 0) -> str:
            idx = (seed + salt + rng.randint(0, 10_000)) % len(options)
            return options[idx]

        def pick_many(options: List[str], count: int) -> List[str]:
            shuffled = list(options)
            rng.shuffle(shuffled)
            return shuffled[: max(0, min(count, len(shuffled)))]

        cited_authors_raw = [c.author for c in citations if c.author]
        cited_authors: List[str] = []
        seen_authors = set()
        for author in cited_authors_raw:
            inline = self._inline_author_attribution(author or "")
            key = self.preprocessor.normalize(inline).lower()
            if not inline or key in seen_authors:
                continue
            seen_authors.add(key)
            cited_authors.append(inline)
            if len(cited_authors) >= 2:
                break
        if cited_authors:
            pair = f"{cited_authors[0]}" + (f" и {cited_authors[1]}" if len(cited_authors) > 1 else "")
            fathers_line = pick(
                [
                    f"Опыт Церкви, раскрытый в слове {pair}, напоминает: духовная жизнь требует не теории, а постоянного внутреннего делания.",
                    f"Святоотеческое наследие, раскрытое в слове {pair}, учит нас хранить трезвение и верность Христу в малых ежедневных решениях.",
                    f"Через слово {pair} Церковь вновь и вновь показывает: путь спасения проходит через смирение, молитву и деятельную любовь.",
                ],
                salt=11,
            )
        else:
            fathers_line = pick(
                [
                    "Святоотеческая традиция напоминает: духовная жизнь требует не внешней формы, а постоянного внутреннего делания перед Богом.",
                    "Опыт святых отцов говорит нам, что вера крепнет тогда, когда человек хранит сердце от ожесточения и ежедневно трудится над собой.",
                    "Церковное предание учит нас: без покаянной работы над собой и без милосердия к ближним духовная жизнь быстро вырождается в привычку.",
                ],
                salt=12,
            )

        topic_low = topic.lower()
        feast_sub = self._feast_subtopic(topic_low)
        event_profile = self._event_profile(topic_low)
        sin_profile = self._sin_profile(topic_low)
        is_sin_topic = sin_profile is not None or self._is_sin_topic_low(topic_low)
        sin_name_nominative = str(sin_profile.get("title_topic", "грех и страсть")).lower() if sin_profile else "грех и страсть"
        sin_name_genitive = str(sin_profile.get("name_genitive", "греха и нераскаянной страсти")) if sin_profile else "греха и нераскаянной страсти"
        sin_focus = str(sin_profile.get("focus", "он отделяет человека от Бога и лишает душу мира")) if sin_profile else "он отделяет человека от Бога и лишает душу мира"
        sin_practice = str(
            sin_profile.get(
                "practice",
                "честно исповедовать грех, отвергать самооправдание и последовательно исправлять жизнь по Евангелию",
            )
        ) if sin_profile else "честно исповедовать грех, отвергать самооправдание и последовательно исправлять жизнь по Евангелию"
        if is_sin_topic:
            fathers_line = pick(
                [
                    "Святоотеческая традиция строго напоминает: покаяние не может быть формальным. Пока человек не отвергнет грех делом, сердце не обретет мира.",
                    "Опыт Церкви учит: борьба со страстью требует трезвения, дисциплины и постоянной молитвы, а не только краткого эмоционального порыва.",
                    "Через слово святых отцов Церковь вновь и вновь показывает: свобода от греха приходит через честную исповедь, смирение и терпеливый духовный труд.",
                ],
                salt=40,
            )
        if not occasion_provided:
            occasion = pick(
                [
                    "воскресной литургии",
                    "церковной молитвы",
                    "дня Господня",
                    "приходского богослужения",
                ],
                salt=7,
            )
        if self._is_lazarus_topic(req):
            doctrinal = pick(
                [
                    "Лазарева суббота открывает нам одно из самых глубоких евангельских свидетельств о Христе: у гроба Лазаря Господь являет и Свою человеческую скорбь, и Божественную власть над смертью. Это событие утверждает веру Церкви в то, что Христос есть Воскресение и Жизнь.",
                    "Воспоминание о воскрешении праведного Лазаря в Вифании напоминает нам: Господь приходит даже туда, где, по человеческому суду, все уже окончательно потеряно. Для Бога нет безнадежных состояний души, если человек открывается вере и покаянию.",
                    "Лазарева суббота стоит на пороге Страстной седмицы как живое знамение победы Христа над смертью. Церковь показывает нам, что путь к Пасхе проходит через веру, терпение и надежду на Господа, Который выводит человека из духовной тьмы к свету жизни.",
                ]
            )
            practice = pick(
                [
                    "В этот день важно посмотреть вглубь своего сердца: где в нас омертвели молитва, сострадание и ревность о Боге. Будем просить Господа, чтобы Он, как Лазаря, воззвал и нашу душу к новой жизни через исповедь, причастие и дела милосердия.",
                    "Лазарева суббота учит нас не отчаиваться за себя и за близких. Даже если человек долго остается в духовной холодности, Христос силен оживить его, когда мы соединяем молитву, терпение и деятельную любовь.",
                    "Перед входом в дни Страстной седмицы примем решение жить внимательнее: меньше осуждать, чаще молиться, хранить мир в семье и помогать нуждающимся. Так память о Лазаре станет для нас не только историей, но началом личного духовного пробуждения.",
                ],
                salt=1,
            )
        elif self._is_prodigal_topic(req):
            doctrinal = pick(
                [
                    "Притча о блудном сыне раскрывает тайну Божией любви: Господь не отвергает кающегося, но принимает его как сына и возвращает ему утраченное достоинство. В центре притчи стоит не только падение человека, но прежде всего милосердие Отца.",
                    "В евангельской притче о блудном сыне мы видим путь души: удаление от Отца, горечь внутреннего разорения и спасительное решение вернуться домой. Эта дорога покаяния остается живой для каждого христианина.",
                    "Притча о блудном сыне учит нас, что грех всегда ведет к духовному голоду и одиночеству, а покаяние возвращает человека к дому Отца, к миру совести и радости общения с Богом.",
                ]
            )
            practice = pick(
                [
                    "Начнем с малого покаянного шага: перестанем оправдывать себя, честно признаем свои ошибки и попросим у Бога силы исправить жизнь. Так возвращение к Отцу становится реальностью, а не красивым словом.",
                    "Эта притча призывает нас не только каяться, но и учиться прощать. Важно не уподобляться старшему сыну в ожесточении, а радоваться каждому, кто возвращается к Богу, и самим хранить милосердное сердце.",
                    "Будем чаще обращаться к исповеди, беречь молитву и внимательно относиться к ближним, особенно к тем, кто оступился. Милость, которую мы сами получаем от Бога, должна становиться и нашим отношением к людям.",
                ],
                salt=1,
            )
        elif event_profile is not None:
            event_name = str(event_profile.get("name", topic))
            focus = str(event_profile.get("focus", "Господь призывает нас к живой вере и покаянию"))
            event_practice = str(event_profile.get("practice", "хранить молитву, мир и деятельную любовь"))
            doctrinal = pick(
                [
                    f"Церковное воспоминание {event_name} открывает перед нами евангельский смысл: {focus}.",
                    f"Тема «{event_name}» напоминает нам, что в центре христианской жизни стоит не внешняя форма, а встреча с живым Христом. Через это событие Церковь учит нас, что {focus}.",
                    f"В событии {event_name} мы видим путь спасения для каждого человека: {focus}.",
                ]
            )
            practice = pick(
                [
                    f"Практический вывод для нас очевиден: {event_practice}.",
                    f"Если мы хотим, чтобы это евангельское слово стало жизнью, нам важно {event_practice}.",
                    f"Пусть тема «{event_name}» побуждает нас не к отвлеченным рассуждениям, а к конкретному деланию: {event_practice}.",
                ],
                salt=1,
            )
        elif self._is_resurrection_topic(req):
            doctrinal = pick(
                [
                    "Воскресение Христово открывает нам не просто память о событии, а новую реальность: смерть уже не имеет последнего слова, потому что Христос победил ад и даровал человеку путь к вечной жизни.",
                    "Пасхальная весть говорит каждому сердцу: Бог не оставил человека в плену тьмы, но Сам вошел в глубину человеческой боли, чтобы вывести нас к свету воскресения и надежды.",
                    "Тайна Воскресения учит нас, что Божия любовь сильнее греха, страха и отчаяния, а значит, даже в самых тяжелых обстоятельствах христианин может жить надеждой и мужеством веры.",
                    "Воскресение Христово свидетельствует, что история человека не заканчивается могилой: Господь дарует нам не временное утешение, а подлинную победу жизни над смертью.",
                ]
            )
            practice = pick(
                [
                    "Будем хранить пасхальную радость не только в словах, но и в делах: примиряться с ближними, поддерживать тех, кто в скорби, и благодарить Бога за каждый день как за дар новой жизни.",
                    "Пусть вера в Воскресшего Христа выражается в конкретной заботе о семье, в терпении к немощам друг друга и в милосердии к тем, кто нуждается в нашем времени и участии.",
                    "Если Христос воскрес, значит, и наша повседневность может быть преображена: оставим уныние, будем внимательны к молитве и станем носителями мира там, где прежде было раздражение и холодность.",
                    "Пасхальная вера призывает нас жить как люди надежды: не поддаваться отчаянию, хранить благодарность Богу и нести свет воскресшей любви в отношения с ближними.",
                ],
                salt=1,
            )
        elif any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            doctrinal = pick(
                [
                    "Церковь именует Пресвятую Богородицу Честнейшей Херувим и Славнейшей без сравнения Серафим. В Ее смиренном согласии на волю Божию мы видим образ совершенного доверия Богу и послушания Евангелию.",
                    "Тайна Боговоплощения неразрывно связана с Пресвятой Богородицей: через Ее кротость и веру Сын Божий приходит в мир ради спасения человека. Поэтому почитание Богородицы всегда ведет нас ко Христу, а не уводит от Него.",
                    "Образ Божией Матери учит нас чистоте сердца, тишине внутренней молитвы и терпению в скорбях. В Ней Церковь созерцает Материнское заступничество и надежду для каждого кающегося.",
                ]
            )
            practice = pick(
                [
                    "Будем чаще прибегать к молитве Пресвятой Богородице, особенно когда сердце смущено и ослаблено. Ее материнское предстательство помогает человеку не впадать в уныние и возвращает душу к миру.",
                    "Почитание Богородицы должно выражаться не только в словах, но и в жизни: хранить целомудрие мыслей, беречь язык от осуждения и учиться смирению в семье и на работе.",
                    "Когда мы обращаемся к Божией Матери с верой, важно соединять молитву с добрыми делами, прощением и милосердием. Так сердце становится способным принять благодать Христову.",
                ],
                salt=1,
            )
        elif self._is_feast_topic(topic_low):
            if feast_sub == "trinity":
                doctrinal = pick(
                    [
                        "День Святой Троицы открывает тайну жизни Церкви как жизни в Святом Духе. Пятидесятница свидетельствует, что Господь не оставляет учеников, но подает силу благодати для проповеди, молитвы и единства.",
                        "В праздник Троицы мы исповедуем, что Дух Святой оживотворяет Церковь, собирает верных в любовь и дает человеку способность жить по Евангелию не своими силами, а помощью Божией.",
                        "Пятидесятница напоминает: христианская жизнь не сводится к правилам, она есть участие в благодати Духа Святого, Который освящает ум, сердце и волю человека.",
                    ]
                )
                practice = pick(
                    [
                        "Будем просить Духа Святого о просвещении ума и очищении сердца: чтобы в семье было больше терпения, в словах - больше мира, а в делах - больше милосердия.",
                        "Праздник Троицы призывает нас беречь церковное единство, избегать осуждения и сохранять мир с ближними, потому что плод Духа - любовь, радость и кротость.",
                        "Подлинно почтить День Святой Троицы значит жить внимательнее к молитве, чаще прибегать к Таинствам и в каждом дне искать волю Божию, а не свою только правоту.",
                    ],
                    salt=1,
                )
            elif feast_sub == "entry_jerusalem":
                doctrinal = pick(
                    [
                        "Вход Господень в Иерусалим раскрывает парадокс Царства Христова: Спаситель входит не в земной славе, а в смирении, принимая путь Креста ради спасения мира.",
                        "Вербное воскресенье напоминает, что легко восклицать Христу осанну, но труднее оставаться Ему верными в дни испытаний. Церковь зовет нас к постоянству, а не к краткому воодушевлению.",
                        "Праздник Входа Господня учит нас встречать Христа в глубине сердца, чтобы внешнее торжество сопровождалось внутренним покаянием и готовностью следовать за Ним.",
                    ]
                )
                practice = pick(
                    [
                        "Будем встречать Господа не только ветвями в храме, но и делами примирения, вниманием к слабым и готовностью отвергнуть греховные привычки.",
                        "Перед Страстной седмицей постараемся хранить молитвенную собранность, меньше спорить, чаще прощать и поддерживать тех, кто рядом переживает скорби.",
                        "Пусть верность Христу проявится в простых решениях дня: говорить правду с любовью, не отвечать раздражением на раздражение и помнить о заповедях.",
                    ],
                    salt=1,
                )
            else:
                doctrinal = pick(
                    [
                        f"Церковный праздник «{topic}» раскрывает перед нами спасительное действие Божие в истории и в личной жизни человека. Через богослужение и молитву Церковь вводит нас в живой опыт встречи со Христом.",
                        f"Праздник «{topic}» напоминает, что вера не ограничивается знанием событий: Господь призывает нас принимать благодать в сердце и преображать повседневную жизнь по Евангелию.",
                        f"В дни праздника «{topic}» Церковь учит нас смотреть на жизнь в свете Царства Божия: видеть в каждом дне не только трудности, но и путь к спасению через верность Христу.",
                    ]
                )
                practice = pick(
                    [
                        "Подлинное празднование начинается не только в храме, но продолжается дома: в мире с ближними, в благодарении Богу и в делах милосердия.",
                        "Постараемся встретить праздник с очищенным сердцем: примириться с обиженными, восстановить молитвенное правило и внимательнее относиться к своим словам и поступкам.",
                        "Будем беречь дар церковного праздника в течение всей недели: читать Евангелие, хранить благодарность и помогать тем, кто рядом нуждается в поддержке.",
                    ],
                    salt=1,
                )
        elif self._is_saint_topic(topic_low):
            doctrinal = pick(
                [
                    f"Память {topic} напоминает нам, что святость рождается не в особых обстоятельствах, а в ежедневной верности Богу. Житие святого показывает, как молитва, смирение и любовь преображают человеческую жизнь.",
                    f"Церковь прославляет {topic} не только ради исторического воспоминания, но как живой пример для нас: каждый христианин призван подражать святости в меру своих сил и обстоятельств.",
                    f"Обращаясь к памяти {topic}, мы видим, что путь ко Христу проходит через терпение, покаяние и милосердие. Именно так святые становились светильниками для Церкви и народа Божия.",
                ]
            )
            practice = pick(
                [
                    f"Будем просить у Бога, чтобы по молитвам {topic} Господь укрепил нас в вере, избавил от духовной расслабленности и научил жить по совести.",
                    "Подражание святому начинается с малого: хранить молитву, не оправдывать свои страсти и учиться деятельной любви к ближнему в обычных делах дня.",
                    "Пусть память святого станет для нас не внешним почитанием, а решимостью менять жизнь: чаще исповедоваться, внимательнее относиться к слову и творить добро ради Христа.",
                ],
                salt=1,
            )
        elif is_sin_topic:
            doctrinal = pick(
                [
                    f"Сегодня Церковь прямо говорит о тяжести {sin_name_genitive}: {sin_focus}. Пока человек оправдывает этот грех, он постепенно теряет трезвение, молитвенную силу и внутренний мир.",
                    f"Святоотеческая традиция предупреждает: нераскаянный грех укореняется и становится духовной болезнью. В отношении {sin_name_genitive} это особенно очевидно: страсть сначала обещает легкость, но затем порабощает сердце.",
                    "Евангелие не оставляет нас в безнадежности: даже из глубины падения можно выйти, если человек перестает прикрывать страсть самооправданием и начинает путь решительного покаяния.",
                ]
            )
            practice = pick(
                [
                    f"Исправление невозможно без конкретного подвига: {sin_practice}. Это не разовое решение, а ежедневная духовная дисциплина перед Богом.",
                    f"Если мы всерьез хотим освободиться от {sin_name_genitive}, нужно действовать без промедления: честная исповедь, отсечение поводов ко греху, усиленная молитва и постоянный контроль совести.",
                    f"Покаянный труд всегда практичен: не откладывать на завтра, не оправдывать себя и не играть с тем, что уже ранило душу. Так постепенно исцеляется сердце и возвращается свобода во Христе.",
                ],
                salt=1,
            )
        elif any(w in topic_low for w in ["любов", "милосерд", "ближн", "прощ"]):
            doctrinal = pick(
                [
                    "Любовь в христианском понимании — это не только чувство, а жертвенное делание, в котором человек учится видеть в ближнем образ Божий и служить ему ради Христа.",
                    "Господь открывает нам, что мера духовной зрелости определяется не громкостью слов, а способностью терпеть, прощать и нести тяготы друг друга.",
                    "Там, где любовь соединяется со смирением, исчезает вражда и рождается подлинная церковная общность, в которой каждый поддерживает другого на пути спасения.",
                ]
            )
            practice = pick(
                [
                    "Постараемся в повседневной жизни говорить мягче, слушать внимательнее и не отвечать злом на зло. Через эти простые шаги сердце учится евангельской любви.",
                    "Будем учиться милосердию в делах: поддержать одинокого, навестить больного, помочь нуждающемуся. Так любовь перестает быть красивой идеей и становится дыханием нашей веры.",
                    "Если нам трудно простить, начнем хотя бы с молитвы за обидевшего. Благодать Божия постепенно смягчает сердце и делает возможным то, что вчера казалось недостижимым.",
                ],
                salt=1,
            )
        else:
            doctrinal = pick(
                [
                    "Евангелие призывает нас не к внешнему благочестию, а к внутреннему преображению сердца. Когда человек доверяет Богу, он получает силы идти путём веры даже среди скорбей и сомнений.",
                    "Христианская жизнь начинается с верности Богу в повседневности: в слове, в мыслях, в отношениях с людьми. Там, где есть смирение и молитва, Господь дает человеку крепость духа и ясность пути.",
                    "Святое Писание открывает нам, что вера становится живой тогда, когда соединяется с любовью и делом. Без внутренней перемены сердца духовная жизнь быстро превращается в пустую форму.",
                ]
            )
            practice = pick(
                [
                    "Постараемся ежедневно находить время для молитвы, чтения Евангелия и дела милосердия. В этих шагах постепенно раскрывается настоящая христианская зрелость.",
                    "Будем хранить внимание к совести, избегать осуждения и чаще обращаться к Таинствам Церкви. Каждое доброе дело, совершенное ради Христа, укрепляет душу.",
                    "Пусть в нашем дне будет место для тишины перед Богом, для терпения в семье и для сострадания к нуждающимся. Так вера становится образом жизни, а не только словом.",
                ],
                salt=1,
            )

        if self._is_lazarus_topic(req):
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, Церковь приводит нас в Вифанию, к гробу праведного Лазаря, чтобы укрепить наше сердце перед Страстной седмицей и показать, что Христос действительно властен над смертью.",
                        f"В день {occasion} мы слышим евангельское повествование о Лазаре Четверодневном. Для {audience} это не только память о прошлом, но живой призыв к вере, покаянию и надежде на Господа.",
                        "Лазарева суббота стоит на пороге великих дней страданий и Воскресения Христова, и потому сегодня Церковь особенно ясно говорит нам о силе Божией любви, побеждающей тление и страх.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        "Сегодня Церковь приводит нас в Вифанию, к гробу праведного Лазаря, чтобы укрепить наше сердце перед Страстной седмицей и показать, что Христос действительно властен над смертью.",
                        f"Мы слышим евангельское повествование о Лазаре Четверодневном. Для {audience} это не только память о прошлом, но живой призыв к вере, покаянию и надежде на Господа.",
                        "Лазарева суббота стоит на пороге великих дней страданий и Воскресения Христова, и потому сегодня Церковь особенно ясно говорит нам о силе Божией любви, побеждающей тление и страх.",
                    ]
                )
        elif self._is_prodigal_topic(req):
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, Церковь предлагает нам притчу о блудном сыне как зеркало нашей собственной жизни, чтобы мы увидели путь возвращения к Отцу Небесному.",
                        f"В день {occasion} притча о блудном сыне обращена к {audience} как призыв к покаянию, надежде и восстановлению живой связи с Богом.",
                        "Евангельская история о блудном сыне напоминает: сколько бы человек ни удалялся от Бога, дверь отчего дома остается открытой для кающегося сердца.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        "Сегодня Церковь предлагает нам притчу о блудном сыне как зеркало нашей собственной жизни, чтобы мы увидели путь возвращения к Отцу Небесному.",
                        f"Притча о блудном сыне обращена к {audience} как призыв к покаянию, надежде и восстановлению живой связи с Богом.",
                        "Евангельская история о блудном сыне напоминает: сколько бы человек ни удалялся от Бога, дверь отчего дома остается открытой для кающегося сердца.",
                    ]
                )
        elif event_profile is not None:
            event_name = str(event_profile.get("name", topic))
            focus = str(event_profile.get("focus", "Господь зовет нас к вере и покаянию"))
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, Церковь обращает наш взор к теме «{event_name}» и напоминает нам, что {focus}.",
                        f"В день {occasion} евангельская тема «{event_name}» звучит для нашей общины как личный призыв к духовному обновлению: {focus}.",
                        f"Сегодняшнее церковное слово на тему «{event_name}» открывает для нас путь христианской жизни: {focus}.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        f"Сегодня Церковь обращает наш взор к теме «{event_name}» и напоминает нам, что {focus}.",
                        f"Евангельская тема «{event_name}» звучит для нашей общины как личный призыв к духовному обновлению: {focus}.",
                        f"Церковное слово на тему «{event_name}» открывает для нас путь христианской жизни: {focus}.",
                    ]
                )
        elif self._is_feast_topic(topic_low):
            if occasion_provided:
                if feast_sub == "trinity":
                    intro_body = pick(
                        [
                            f"Сегодня, в день {occasion}, Церковь молитвенно празднует Пятидесятницу и напоминает нам о сошествии Святого Духа на апостолов, через которое раскрылась жизнь Церкви.",
                            f"В день {occasion} тема Святой Троицы звучит для {audience} как призыв открыть сердце действию благодати Духа Святого.",
                            "Праздник Троицы обращает нас к самой глубине церковной жизни: к единству в любви, молитве и верности Христу.",
                        ]
                    )
                elif feast_sub == "entry_jerusalem":
                    intro_body = pick(
                        [
                            f"Сегодня, в день {occasion}, Церковь вспоминает Вход Господень в Иерусалим и зовет нас встретить Христа не только внешней радостью, но и внутренним покаянием.",
                            f"В день {occasion} Вербное воскресенье напоминает {audience}, что верность Христу проверяется не в торжественных словах, а в ежедневном следовании Евангелию.",
                            "Праздник Входа Господня ставит перед нами важный вопрос: готовы ли мы идти за Христом не только в дни славы, но и на пути Креста.",
                        ]
                    )
                else:
                    intro_body = pick(
                        [
                            f"Сегодня, в день {occasion}, Церковь празднует «{topic}» и приглашает нас не только вспомнить событие, но и принять его духовный смысл как руководство для жизни.",
                            f"В день {occasion} тема «{topic}» звучит для {audience} как живой призыв к благодарению Богу, молитве и обновлению сердца.",
                            f"Праздник «{topic}» напоминает нам, что Господь действует в истории спасения и в нашей личной жизни, если мы открываемся Его благодати с верой и смирением.",
                        ]
                    )
            else:
                if feast_sub == "trinity":
                    intro_body = pick(
                        [
                            "Сегодня Церковь молитвенно празднует Пятидесятницу и напоминает нам о сошествии Святого Духа на апостолов, через которое раскрылась жизнь Церкви.",
                            f"Тема Святой Троицы звучит для {audience} как призыв открыть сердце действию благодати Духа Святого.",
                            "Праздник Троицы обращает нас к самой глубине церковной жизни: к единству в любви, молитве и верности Христу.",
                        ]
                    )
                elif feast_sub == "entry_jerusalem":
                    intro_body = pick(
                        [
                            "Сегодня Церковь вспоминает Вход Господень в Иерусалим и зовет нас встретить Христа не только внешней радостью, но и внутренним покаянием.",
                            f"Вербное воскресенье напоминает {audience}, что верность Христу проверяется не в торжественных словах, а в ежедневном следовании Евангелию.",
                            "Праздник Входа Господня ставит перед нами важный вопрос: готовы ли мы идти за Христом не только в дни славы, но и на пути Креста.",
                        ]
                    )
                else:
                    intro_body = pick(
                        [
                            f"Сегодня Церковь празднует «{topic}» и приглашает нас не только вспомнить событие, но и принять его духовный смысл как руководство для жизни.",
                            f"Тема «{topic}» звучит для {audience} как живой призыв к благодарению Богу, молитве и обновлению сердца.",
                            f"Праздник «{topic}» напоминает нам, что Господь действует в истории спасения и в нашей личной жизни, если мы открываемся Его благодати с верой и смирением.",
                        ]
                    )
        elif self._is_saint_topic(topic_low):
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, Церковь обращает наш взор к памяти {topic}, чтобы через пример святого укрепить нас в вере, надежде и любви.",
                        f"В день {occasion} мы вспоминаем {topic}. Для {audience} это не только память о прошлом, но призыв к живому подражанию христианской верности в повседневности.",
                        f"Память {topic} напоминает нам: святость возможна и в земной жизни, когда человек с постоянством выбирает Евангелие, молитву и милосердие.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        f"Сегодня Церковь обращает наш взор к памяти {topic}, чтобы через пример святого укрепить нас в вере, надежде и любви.",
                        f"Мы вспоминаем {topic}. Для {audience} это не только память о прошлом, но призыв к живому подражанию христианской верности в повседневности.",
                        f"Память {topic} напоминает нам: святость возможна и в земной жизни, когда человек с постоянством выбирает Евангелие, молитву и милосердие.",
                    ]
                )
        elif is_sin_topic:
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, Церковь строго и трезвенно говорит о теме {sin_name_genitive}, потому что этот грех разрушает душу изнутри и лишает человека мира с Богом.",
                        f"В день {occasion} нам необходимо без самооправдания увидеть, как именно действует в нас эта страсть, и что мы готовы изменить ради Христа.",
                        f"Сегодняшнее слово, посвященное теме {sin_name_genitive}, обращено ко всей церковной общине как призыв к решительному покаянию: нельзя одновременно служить Богу и беречь страсть.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        f"Сегодня Церковь строго и трезвенно говорит о теме {sin_name_genitive}, потому что этот грех разрушает душу изнутри и лишает человека мира с Богом.",
                        "Нам необходимо без самооправдания увидеть, как именно действует в нас эта страсть, и что мы готовы изменить ради Христа.",
                        f"Слово, посвященное теме {sin_name_genitive}, обращено ко всей церковной общине как призыв к решительному покаянию: нельзя одновременно служить Богу и беречь страсть.",
                    ]
                )
        else:
            if occasion_provided:
                intro_body = pick(
                    [
                        f"Сегодня, в день {occasion}, обратимся к размышлению на тему «{topic}». Эта тема касается каждого из нас, потому что именно в обычных обстоятельствах раскрывается подлинная глубина веры.",
                        f"В день {occasion} Церковь вновь напоминает нам о теме «{topic}». Для {audience} это не отвлеченное рассуждение, а живой вопрос духовного пути и ответственности перед Богом.",
                        f"Обращаясь к теме {topic}, будем помнить: Господь ждет от нас не красивых слов, а реального движения сердца к Нему. Именно так человек духовно взрослеет и укрепляется в истине.",
                    ]
                )
            else:
                intro_body = pick(
                    [
                        f"Сегодня обратимся к размышлению на тему «{topic}». Эта тема касается каждого из нас, потому что именно в обычных обстоятельствах раскрывается подлинная глубина веры.",
                        f"Церковь вновь напоминает нам о теме «{topic}». Для {audience} это не отвлеченное рассуждение, а живой вопрос духовного пути и ответственности перед Богом.",
                        f"Обращаясь к теме {topic}, будем помнить: Господь ждет от нас не красивых слов, а реального движения сердца к Нему. Именно так человек духовно взрослеет и укрепляется в истине.",
                    ]
                )
        intro_body = intro_body.replace("Для приход это", "Для прихода это")
        intro = "Во имя Отца, и Сына, и Святого Духа!\nДорогие братья и сестры!\n" + intro_body
        if self._is_lazarus_topic(req):
            intro_extension_pool = [
                "Это евангельское событие дано нам не только для воспоминания, но и для личного духовного пробуждения перед святыми днями.",
                "Попросим Господа, чтобы слово о Лазаре коснулось не только нашего слуха, но и глубины сердца.",
                "Пусть сегодняшняя молитва укрепит нас в надежде, что для Христа нет безнадежных состояний души.",
                "Церковь говорит об этом ради нашего исцеления, чтобы мы научились жить с упованием и благодарением.",
            ]
        elif self._is_prodigal_topic(req):
            intro_extension_pool = [
                "Постараемся услышать эту притчу как личное слово Божие, обращенное к нашей совести именно сегодня.",
                "В каждом из нас есть и блуждающий сын, и призвание вернуться в объятия милосердного Отца.",
                "Пусть это евангельское слово научит нас не отчаиваться в своих падениях и не осуждать возвращающихся.",
                "Церковь напоминает об этой притче, чтобы укрепить нас в надежде и решимости начать путь покаяния.",
            ]
        elif event_profile is not None:
            event_focus = str(event_profile.get("focus", "Господь зовет нас к вере и покаянию"))
            event_practice = str(event_profile.get("practice", "хранить молитву и милосердие"))
            intro_extension_pool = [
                f"Это слово дано нам не только для знания, но и для внутренней перемены: {event_focus}.",
                f"Примем услышанное как личное обращение Божие и начнем действовать: {event_practice}.",
                "Пусть сегодняшнее евангельское напоминание коснется глубины сердца и станет началом живого делания.",
                "Церковь предлагает нам этот образ ради исцеления души и укрепления надежды в Боге.",
            ]
        elif self._is_resurrection_topic(req):
            intro_extension_pool = [
                "Пасхальная весть требует не только радостного возгласа, но и внутренней перемены всей жизни.",
                "Вслушаемся в церковное слово так, чтобы оно стало для нас источником мужества, мира и веры.",
                "Пусть радость о Воскресшем Христе озарит и нашу молитву, и наши отношения с ближними.",
                "Сегодня особенно важно открыть сердце благодати, которая преображает человека изнутри.",
            ]
        elif self._is_feast_topic(topic_low):
            intro_extension_pool = [
                "Праздник Церкви становится плодотворным тогда, когда переходит из храма в повседневность наших решений.",
                "Пусть сегодняшнее слово поможет нам увидеть, как богослужение направляет нас к реальной христианской жизни.",
                "Церковь предлагает нам не только память о событии, но и путь практического преображения сердца.",
                "Будем внимательны, чтобы благодать праздника не осталась внешним впечатлением, а стала содержанием нашей жизни.",
            ]
        elif self._is_saint_topic(topic_low):
            intro_extension_pool = [
                "Память святого дана нам как ориентир, чтобы мы учились верности Богу в обычных обстоятельствах.",
                "Подлинное почитание святых раскрывается в подражании их молитве, терпению и любви к людям.",
                "Пусть пример угодника Божия станет для нас живым призывом к покаянию и духовной решимости.",
                "Церковь напоминает нам о святых, чтобы мы не теряли надежды и мужества на собственном пути ко Христу.",
            ]
        elif is_sin_topic:
            intro_extension_pool = [
                f"Не уклонимся в общие слова: сегодня требуется честное обличение {sin_name_genitive} и мужество назвать вещи своими именами.",
                "Строгость церковного слова - это не жестокость, а врачевство, которое отрезвляет душу и возвращает человека к жизни.",
                "Без покаяния грех укореняется и становится привычкой, поэтому важно не медлить с внутренним исправлением.",
                "Примем это наставление как Божий зов к обновлению сердца, пока совесть еще слышит голос Евангелия.",
            ]
        else:
            intro_extension_pool = [
                "Пусть сегодняшнее слово станет для нас не только размышлением, но и началом конкретного шага к Богу.",
                "Вслушаемся в это слово так, чтобы оно коснулось не только ума, но и сердца.",
                "Церковь говорит об этом не отвлеченно, а ради нашего духовного исцеления и мира в душе.",
                "Постараемся принять это наставление как личное обращение Божие к нашей совести и нашей ответственности.",
            ]
        intro += " " + " ".join(pick_many(intro_extension_pool, 3))

        first_main = (
            f"Евангельский фрагмент {bible_ref} направляет нас к живой вере и упованию на Господа. {doctrinal}"
            if bible_ref
            else doctrinal
        )
        bridge = pick(
            [
                "Если же мы падаем, не будем отчаиваться: Господь поднимает кающегося и укрепляет его на новом пути.",
                "Даже после падений будем возвращаться к покаянию и молитве: милость Божия сильнее нашей немощи.",
                "Путь спасения совершается не в безошибочности, а в верном возвращении к Богу после каждого падения.",
                "Когда сердце ослабевает, не оставим молитву: Господь укрепляет тех, кто ищет Его с терпением и надеждой.",
            ],
            salt=20,
        )
        third_main = practice + " " + bridge
        if any(x in audience_low for x in ["молод", "студент", "подрост"]):
            fourth_main = pick(
                [
                    "Особенно важно сказать об этом молодежи: духовная жизнь строится не мгновенно, а через верность в малом. Когда человек учится хранить чистоту мысли, уважение к родителям и ответственность в труде, вера становится прочным основанием жизни.",
                    "Для молодых людей этот путь особенно важен: в мире шума и соблазнов нужно хранить трезвение ума, чистоту сердца и верность молитве, чтобы не потерять главное.",
                    "Молодежи Церковь напоминает: будущее строится не только знаниями и карьерой, но и совестью, молитвой и благоговением перед Богом.",
                ],
                salt=22,
            )
        elif any(x in audience_low for x in ["сем", "родител", "супруг"]):
            fourth_main = pick(
                [
                    "Для семейной жизни эта тема имеет особую силу: дом становится по-настоящему христианским там, где есть совместная молитва, взаимное прощение и готовность нести тяготы друг друга.",
                    "В семье вера проверяется ежедневно: в терпении к слабостям ближнего, в мягком слове и в готовности первым идти к примирению.",
                    "Супружеская и родительская любовь крепнет там, где есть смирение, благодарность Богу и честная забота друг о друге.",
                ],
                salt=23,
            )
        else:
            fourth_main = pick(
                [
                    "Проверяйте свои решения светом Евангелия: ведут ли они к миру, правде и любви Христовой. Так даже простые обязанности дня становятся частью духовного пути.",
                    "Не отделяйте веру от повседневности: именно в обычных разговорах, трудах и отношениях с людьми проявляется подлинное христианство.",
                    "Будем помнить, что духовная жизнь совершается в простых делах дня: в верности слову, в терпении и в милосердии к ближнему.",
                    "Если наше сердце хранит молитву и благодарение, то и повседневные заботы становятся местом встречи с благодатью Божией.",
                ],
                salt=24,
            )

        scripture_transition = pick(
            [
                "Именно здесь становится видно, что библейское слово обращено не к отвлеченной теории, а к нашему ежедневному выбору между покаянием и самооправданием.",
                "Эти свидетельства Писания и церковного опыта важны для нас только тогда, когда мы принимаем их как руководство к конкретной перемене жизни.",
                "От богословского смысла необходимо перейти к внутреннему деланию: услышанное слово должно изменить не только мысли, но и привычки сердца.",
            ],
            salt=25,
        )
        if is_sin_topic:
            scripture_transition = pick(
                [
                    "Здесь особенно ясно: речь идет не о формальном согласии с истиной, а о решительном разрыве со страстью и о честном покаянном труде.",
                    "Эти слова Писания обличают нас персонально: невозможно сохранить грех и одновременно ожидать мира совести и живой молитвы.",
                    "Отсюда прямой вывод: пока мы оправдываем грех, духовная жизнь остается поверхностной; только покаяние возвращает сердце к свободе во Христе.",
                ],
                salt=25,
            )

        synthesis_main = pick(
            [
                "Таким образом, богословский смысл темы раскрывается в простом, но трудном правиле: верность Христу проверяется в каждом дне, в каждом слове и в каждом выборе совести.",
                "Итак, центр сегодняшнего слова не в красивой риторике, а в пути внутреннего преображения, который начинается сейчас и продолжается во всех обстоятельствах жизни.",
                "Соберем главное: вера, освященная молитвой и подтвержденная делами милосердия, постепенно делает человека цельным и способным к миру с Богом и ближними.",
            ],
            salt=26,
        )
        if is_sin_topic:
            synthesis_main = pick(
                [
                    "Подведем итог строго: без отсечения страсти и без конкретного покаянного труда тема останется разговором, но не станет путем спасения.",
                    "Итак, духовный вывод ясен: нужно не обсуждать грех издалека, а назвать его в себе, исповедовать и последовательно менять образ жизни по Евангелию.",
                    "Суммируем услышанное: строгость к себе, честность на исповеди и постоянная молитва — это не дополнение, а необходимое основание освобождения от страсти.",
                ],
                salt=26,
            )

        quote_parts = self._build_quote_paragraphs(req, citations)
        main_parts = [first_main]
        if quote_parts:
            main_parts.extend(quote_parts)
            main_parts.append(scripture_transition)
        if rng.random() < 0.95:
            main_parts.append(fathers_line)
        main_parts.append(third_main)
        main_parts.append(fourth_main)
        if self._is_lazarus_topic(req):
            thematic_pool = [
                "Евангелие о Лазаре показывает нам слезы Христа у гроба друга. Эти слезы раскрывают, что Богу не безразлична человеческая боль: Господь не наблюдает со стороны, а входит в нашу скорбь и исцеляет ее Своей любовью.",
                "По слову Спасителя Лазарь выходит из гроба, и этим Церковь заранее свидетельствует о грядущей Пасхе. Перед нами не просто чудо, а откровение о том, Кто стоит перед нами: истинный Бог и истинный Человек, Победитель смерти.",
                "Лазарева суббота напоминает: духовная смерть начинается незаметно, когда мы привыкаем к греху и откладываем покаяние. Но Господь и сегодня зовет человека выйти из внутренней тьмы к свету благодатной жизни.",
                "Обратим внимание на веру Марфы и Марии: среди скорби они не разрывают общения с Господом. Так и мы, встречая испытания, должны не замыкаться в обиде, а приносить свою боль в молитве Христу.",
                "После Лазаревой субботы начинается путь к Кресту и Пасхе. Значит, и наша вера должна стать более собранной: меньше рассеянности, больше молитвы, больше мира в семье и терпения к ближним.",
                "Господь повелевает развязать Лазаря от погребальных пелен, и это образ церковной жизни: Христос оживляет, а община любви помогает человеку освободиться от старых привычек, страстей и уныния.",
                "Праздник Лазаревой субботы дает нам надежду и о наших усопших: мы молимся не в отчаянии, а в вере, что для Христа смерть не является концом человеческой судьбы.",
            ]
        elif self._is_prodigal_topic(req):
            thematic_pool = [
                "Блудный сын приходит в себя среди голода и унижения. Так и в нашей жизни истинное покаяние начинается тогда, когда человек перестает винить обстоятельства и честно смотрит на себя перед Богом.",
                "Отец не устраивает допроса, но выходит навстречу кающемуся сыну. Это образ безмерной Божией милости, которая предваряет нас и поднимает, когда мы только решаем вернуться.",
                "Притча о блудном сыне предупреждает нас и о духовной болезни старшего брата: внешняя правильность без любви делает сердце жестким и неспособным к состраданию.",
                "Возвращение к Отцу требует конкретных шагов: оставить привычный грех, примириться с ближними, начать молитвенную жизнь и не откладывать исповедь.",
                "В доме Отца звучит радость о возвращении погибшего. Церковь учит нас радоваться каждому покаянному движению человека и не закрывать дверь милосердия перед теми, кто оступился.",
                "Эта притча помогает нам увидеть, что христианская жизнь - не страх наказания, а путь сыновнего доверия Богу, Который ждет нас с любовью и терпением.",
            ]
        elif event_profile is not None:
            event_name = str(event_profile.get("name", topic))
            event_focus = str(event_profile.get("focus", "Господь зовет нас к вере и обновлению сердца"))
            event_practice = str(event_profile.get("practice", "жить по Евангелию в повседневности"))
            thematic_pool = [
                f"Церковная тема «{event_name}» помогает нам увидеть, что {event_focus}.",
                f"Через событие {event_name} Господь обращается и к нашей совести, призывая нас к покаянию, трезвению и живой вере.",
                f"Для нас это не только историческое воспоминание, но и практическая школа духовной жизни: {event_practice}.",
                f"Когда мы размышляем над темой «{event_name}», становится яснее, что христианство требует не слов только, но постоянства в добром делании.",
                f"Пусть эта евангельская тема станет для нас источником надежды: Бог действует и сегодня, когда человек открывается благодати и старается жить по правде.",
                f"Проверим себя перед лицом этого слова: становимся ли мы внимательнее к молитве, мягче к ближним и честнее перед собственной совестью.",
            ]
        elif any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            thematic_pool = [
                "Созерцая образ Божией Матери, мы учимся не превозноситься, а в тишине сердца говорить Богу: да будет воля Твоя. Именно в таком смирении рождается подлинная свобода от страстей и внутренней суеты.",
                "Почитание Богородицы неразрывно связано с церковной молитвой: через акафисты, каноны и тропари Церковь воспитывает в нас благодарное сердце, благоговение и трезвение ума.",
                "Когда семья хранит молитву к Пресвятой Богородице, в доме становится больше мира и взаимного терпения. Там, где есть кротость и прощение, легче преодолевать и внешние скорби, и внутренние искушения.",
                "Божия Матерь учит нас беречь чистоту сердца: не осуждать ближнего, не ожесточаться в обидах, не искать своей правды любой ценой. Этот путь непрост, но именно он делает душу способной к благодати.",
                "В церковном предании Богородица именуется Скоропослушницей и Утешением скорбящих. Это напоминает нам, что в час испытаний нужно не замыкаться в себе, а с доверием обращаться к Богу и Его Пречистой Матери.",
                "Через богородичные праздники Церковь раскрывает перед нами путь благодарности: видеть в каждом дне не только трудности, но и дары Божии, которые укрепляют веру и побуждают к милосердию.",
            ]
        elif self._is_resurrection_topic(req):
            thematic_pool = [
                "Пасхальная проповедь всегда обращает нас к сердцу Евангелия: Христос разрушил власть смерти и открыл человеку путь к жизни вечной. Поэтому христианская надежда не зависит от обстоятельств, а укоренена в победе Воскресшего Господа.",
                "Смысл пасхальной радости не в кратком эмоциональном подъеме, а в перемене образа жизни: примирение вместо вражды, благодарение вместо ропота, верность заповедям вместо духовной рассеянности.",
                "Когда Церковь поет о Воскресении Христовом, она не вспоминает только прошлое событие, но вводит нас в живой опыт встречи со Спасителем, Который и сегодня действует в Таинствах и молитве.",
                "Воскресение Христово дает мужество переносить скорби: мы знаем, что Господь уже прошел через страдание и смерть и потому может укрепить всякого, кто с верой обращается к Нему.",
                "Пасхальная весть призывает нас быть свидетелями света: в семье, на работе, в общении с людьми хранить мир, кротость и верность правде, чтобы через нас прославлялся Воскресший Христос.",
                "Не будем сводить Пасху только к обряду и внешнему празднику: подлинная пасхальная жизнь начинается там, где человек ежедневно умирает для греха и воскресает для любви и милосердия.",
            ]
        elif self._is_feast_topic(topic_low):
            if feast_sub == "trinity":
                thematic_pool = [
                    "Сошествие Святого Духа на апостолов показывает, что Церковь живет не человеческой силой, а даром Божиим. Там, где есть молитва и смирение, там действует благодать, исцеляющая разделения и страх.",
                    "Праздник Троицы напоминает нам, что Дух Святой освящает не только храм, но и человеческое сердце, когда мы отказываемся от ожесточения, осуждения и самодовольства.",
                    "Единство Церкви строится не на внешнем сходстве, а на любви во Христе. Поэтому в день Пятидесятницы мы особенно призваны хранить мир, слушать друг друга и не разрушать общение гордым словом.",
                    "Жизнь в благодати Духа Святого проверяется в простых вещах: умеем ли мы благодарить, прощать, быть терпеливыми и верными в малом.",
                    "Пусть праздник Троицы научит нас соединять молитву и дело: просить у Бога силы и одновременно трудиться над собой с постоянством и трезвением.",
                    "Дар Святого Духа не отменяет нашего подвига, но делает его возможным: в нем человек получает мужество жить по Евангелию даже среди скорбей и внутренних сомнений.",
                ]
            elif feast_sub == "entry_jerusalem":
                thematic_pool = [
                    "Вход Господень в Иерусалим напоминает нам, что внешняя радость быстро проходит, если сердце не укреплено в покаянии и верности заповедям.",
                    "Толпа встречала Христа как Царя, но немногие были готовы идти за Ним на Голгофу. И сегодня перед каждым из нас стоит тот же выбор: следовать за Спасителем до конца или только пока это удобно.",
                    "Вербное воскресенье призывает нас встречать Христа не шумом эмоций, а тихим решением жить по совести, хранить молитву и бороться со страстями.",
                    "Перед Страстной седмицей важно собраться внутренне: меньше праздных слов, больше тишины перед Богом, больше внимания к боли ближнего.",
                    "Если в нас есть раздражение и гордость, попросим Господа смирить сердце, чтобы встреча с Ним была настоящей, а не только внешней.",
                    "Праздник Входа Господня учит нас мужеству верности: не отступать от Христа, когда путь становится трудным и требует жертвы.",
                ]
            else:
                thematic_pool = [
                    f"Праздник «{topic}» дан нам как школа духовного внимания: в храмовом богослужении мы учимся видеть действие Божие не только в истории, но и в собственной жизни.",
                    f"В церковной памяти «{topic}» соединяются богословие и практика жизни: то, что Церковь воспевает на службе, должно стать содержанием наших мыслей, слов и поступков.",
                    f"Праздничные дни призывают нас к благодарению и трезвению: важно не раствориться во внешней суете, а сохранить внутреннюю тишину, молитву и мир с ближними.",
                    "Подлинная радость праздника проявляется в делах любви: поддержать скорбящего, посетить одинокого, отказаться от осуждения и сохранить добрый дух в семье.",
                    "Литургический ритм Церкви воспитывает в нас постоянство: от праздника к празднику сердце постепенно учится верности Христу и послушанию Его заповедям.",
                    "Будем помнить, что церковный праздник - это не отдых от духовного труда, а благодатное укрепление, чтобы этот труд продолжался в каждом дне.",
                ]
        elif self._is_saint_topic(topic_low):
            thematic_pool = [
                f"Память {topic} раскрывает перед нами конкретный путь святости: верность в молитве, терпение в скорбях и деятельное милосердие к людям.",
                "Жития святых учат нас не искать мгновенных результатов: духовная зрелость рождается в долгом труде над сердцем, в покаянии и смирении.",
                "Святой становится для нас не только историческим образом, но живым ходатаем и наставником, который призывает к ответственности за собственную духовную жизнь.",
                "Подражание святым начинается с простых шагов: хранить язык от осуждения, быть честным в малом, не оставлять молитву даже в усталости.",
                "Когда мы обращаемся к святым в молитве, важно одновременно исправлять жизнь: именно так церковное почитание становится плодотворным и спасительным.",
                f"Через пример {topic} Церковь напоминает, что путь к Богу открыт каждому, кто готов ежедневно выбирать Евангелие вместо самолюбия.",
            ]
        elif is_sin_topic:
            thematic_pool = [
                f"Нужно ясно видеть духовный механизм {sin_name_genitive}: сначала страсть кажется безобидной, потом становится привычкой, а затем начинает управлять человеком.",
                f"Главная опасность {sin_name_genitive} в том, что {sin_focus}. Поэтому молчаливое соглашение с этим грехом постепенно убивает ревность о спасении.",
                "Подлинное покаяние начинается с ответственности: не перекладывать вину на обстоятельства, не объяснять грех характером и не прятаться за внешней религиозностью.",
                "Исповедь должна быть прямой и честной: без туманных формулировок и без самооправдания. Только такая исповедь становится началом реальной перемены жизни.",
                f"В борьбе с темой {sin_name_genitive} нужны конкретные правила жизни: молитва утром и вечером, хранение чувств, отсечение поводов ко греху и регулярный духовный отчет совести.",
                f"Нельзя примиряться с повторяющимся падением в области {sin_name_genitive}. Если грех возвращается, надо усиливать подвиг и просить церковной помощи, а не ждать автоматического исправления.",
                f"Практический путь исправления таков: {sin_practice}. Именно так человек выходит из власти страсти к свободе в Боге.",
                "Строгость к себе должна соединяться с милостью к ближнему: обличая грех в себе, будем беречь другого от унижения и отчаяния.",
            ]
        else:
            thematic_pool = [
                "Духовная жизнь не строится рывками; она созидается ежедневной верностью: в утренней молитве, в честном труде, в бережном отношении к слову и в отказе от осуждения.",
                "Если хотим увидеть плод веры, будем внимательны к малым вещам: как мы разговариваем дома, как относимся к слабости другого, как переносим несправедливость и усталость.",
                "Церковь учит нас соединять молитву и дело: просить помощи Божией и одновременно брать ответственность за поступки, потому что благодать не отменяет нашего труда, а освящает его.",
                "Трезвение начинается с простого вопроса к себе: что сейчас руководит мной - любовь ко Христу или самолюбие? Такой внутренний суд совести постепенно очищает сердце.",
                "В дни смущения особенно важно хранить молитвенное правило хотя бы в малой мере, не прекращать благодарение Богу и помнить, что верность в малом открывает путь к большему.",
                "Господь дает силы не тем, кто никогда не падает, а тем, кто не прекращает подниматься, исповедовать свою немощь и снова искать правды Божией.",
                "Нам полезно чаще обращаться к Евангелию не как к знакомому тексту, а как к живому слову, которое сегодня обращено именно к нашему сердцу.",
            ]
        main_parts.extend(pick_many(thematic_pool, 4))
        if rng.random() < 0.65:
            main_parts.append(
                pick(
                    [
                        "Не будем ждать особых обстоятельств для духовной жизни: именно в обычном дне, в словах и поступках, проявляется наша верность Евангелию.",
                        "Иногда нам кажется, что путь слишком труден, но Господь укрепляет того, кто не оставляет молитву и не теряет надежды.",
                        "Каждое наше малое усилие, совершаемое ради Христа, становится семенем будущего духовного плода.",
                        "Пусть память о Божиих благодеяниях укрепляет нас в благодарности и помогает терпеливо нести жизненные испытания.",
                    ],
                    salt=14,
                )
            )
        if rng.random() < 0.9:
            extra_pool = [
                "Пусть в нашей жизни будет место для молчаливой молитвы, когда мы не требуем от Бога немедленных ответов, а учимся доверять Ему и в ясные, и в трудные дни.",
                "Проверим себя: становимся ли мы мягче сердцем, терпеливее к ближним, внимательнее к заповедям? Именно эти признаки показывают, растет ли в нас духовная жизнь.",
                "Не будем подменять покаяние самооправданием. Лучше честно признать свою немощь и попросить у Господа силы начать снова, чем годами оставаться в одном и том же внутреннем холоде.",
                "Христианская зрелость рождается там, где человек перестает жить только для себя и открывается служению ближнему: словом утешения, делом помощи и жертвой времени.",
                "Когда мы благодарим Бога за малое, сердце становится устойчивее к унынию, а надежда перестает быть абстракцией и превращается в живой опыт присутствия Божия.",
            ]
            if is_sin_topic:
                extra_pool = [
                    f"Проверим себя беспощадно и честно: что именно мы продолжаем оправдывать в области {sin_name_genitive}, хотя совесть уже свидетельствует против этого?",
                    "Не будем прикрывать страсть благочестивыми словами. Бог ждет от нас не объяснений, а реального разрыва с грехом и верности покаянному труду.",
                    f"Даже после падений в теме {sin_name_genitive} не уйдем в отчаяние: снова встанем, исповедуем грех и продолжим борьбу с терпением и надеждой.",
                    "Если сердце ожесточается, нужно усиливать молитву и просить у Бога страха Божия, чтобы грех снова стал для нас грехом, а не привычной нормой.",
                    "Свобода от страсти приходит не сразу, но Господь укрепляет того, кто ежедневно, без самообмана, возвращается к пути исправления.",
                ]
            main_parts.extend(
                pick_many(
                    extra_pool,
                    3,
                )
            )
        main_parts.append(synthesis_main)
        main_parts = [self._dedupe_sentences(p) for p in main_parts]
        main_parts = self._dedupe_paragraphs(main_parts)
        main_parts = self._add_cohesive_transitions(main_parts, is_sin_topic)
        main_parts = self._dedupe_paragraphs([self._dedupe_sentences(p) for p in main_parts])
        if len(main_parts) < 9:
            # Минимум 9 смысловых абзацев в основной части для более насыщенной проповеди.
            refill_pool = pick_many(thematic_pool, 6)
            main_parts.extend(refill_pool)
            main_parts = self._add_cohesive_transitions(main_parts, is_sin_topic)
            main_parts = self._dedupe_paragraphs([self._dedupe_sentences(p) for p in main_parts])
        main = "\n\n".join(main_parts)

        if self._is_lazarus_topic(req):
            conclusion = pick(
                [
                    "В день Лазаревой субботы попросим Господа оживить и наши сердца, чтобы мы встретили Страстную седмицу с покаянием, миром и надеждой. Да укрепит нас Христос в вере и любви. Аминь.",
                    "Будем помнить: Тот, Кто воззвал Лазаря из гроба, силен поднять и нас из всякого духовного падения. Вступим в святые дни с решимостью жить по Евангелию, хранить молитву и творить добро. Аминь.",
                    "Пусть память о Лазаре Четверодневном укрепит нас в уповании на Господа, Который побеждает смерть и дарует новую жизнь кающемуся сердцу. С этой надеждой вступим в путь к Пасхе. Аминь.",
                ],
                salt=2,
            )
        elif self._is_prodigal_topic(req):
            conclusion = pick(
                [
                    "Пусть притча о блудном сыне укрепит нас в решимости встать и идти к Отцу Небесному, не откладывая покаяния и примирения. Господь ждет каждого из нас с любовью. Аминь.",
                    "Будем просить у Бога сердца сыновнего, а не рабского: сердца, которое умеет каяться, прощать и радоваться спасению ближнего. Да укрепит нас в этом Господь. Аминь.",
                    "Не останемся только слушателями притчи: принесем ее в жизнь через исповедь, молитву и милосердие, чтобы радость возвращения к Богу стала нашей личной реальностью. Аминь.",
                ],
                salt=2,
            )
        elif event_profile is not None:
            event_name = str(event_profile.get("name", topic))
            event_practice = str(event_profile.get("practice", "хранить молитву, мир и верность Евангелию"))
            conclusion = pick(
                [
                    f"Пусть тема «{event_name}» укрепит нас в решимости жить по Евангелию: {event_practice}. Аминь.",
                    f"Примем церковное слово на тему «{event_name}» как призыв к обновлению сердца и добрым делам. Да поможет нам Господь {event_practice}. Аминь.",
                    f"Не отложим услышанное на потом: начнем уже сегодня исполнять в жизни то, чему учит нас тема «{event_name}»: {event_practice}. Аминь.",
                ],
                salt=2,
            )
        elif any(w in topic_low for w in ["богород", "пресвят", "дева мар", "матер бож", "владычиц"]):
            conclusion = pick(
                [
                    "Будем просить Пресвятую Богородицу о заступничестве, чтобы Господь даровал нам чистоту сердца, мир в доме и стойкость в вере. Аминь.",
                    "Не ослабеем в молитве к Божией Матери: Ее материнское предстательство помогает нам идти ко Христу путём покаяния и надежды. Аминь.",
                    "Пусть Пречистая Дева укрепит нас в смирении и любви, чтобы в каждом дне мы оставались верными Евангелию Христову. Аминь.",
                ],
                salt=2,
            )
        elif self._is_resurrection_topic(req):
            conclusion = pick(
                [
                    "Будем хранить в сердце пасхальную радость и жить так, чтобы в наших словах и делах отражалась победа Воскресшего Господа над грехом и смертью. Христос Воскресе, дорогие братья и сестры! Воистину Воскресе!",
                    "Не дадим пасхальному свету угаснуть в повседневности: сохраним мир, благодарность и верность Евангелию. Христос Воскресе, дорогие! Воистину Воскресе!",
                    "Пусть сила Воскресения Христова укрепляет нас во всяком добром деле и ведет к жизни вечной. Христос Воскресе, дорогие братья и сестры! Воистину Воскресе!",
                ],
                salt=2,
            )
        elif self._is_feast_topic(topic_low):
            if feast_sub == "trinity":
                conclusion = pick(
                    [
                        "В день Святой Троицы будем просить Господа, чтобы Дух Святой просветил наш ум, согрел сердце и научил нас жить в мире, чистоте и верности Евангелию. Аминь.",
                        "Пусть благодать Святого Духа укрепит нас в церковном единстве, терпении и любви, чтобы каждый день становился для нас путем к Богу. Аминь.",
                        "Примем праздник Пятидесятницы как призыв к внутреннему обновлению: к молитве, трезвению и деятельной любви к ближним. Да поможет нам в этом Господь. Аминь.",
                    ],
                    salt=2,
                )
            elif feast_sub == "entry_jerusalem":
                conclusion = pick(
                    [
                        "Встречая Господа в праздник Входа в Иерусалим, попросим у Него верности и мужества идти за Ним не только в дни радости, но и в дни испытаний. Аминь.",
                        "Пусть Вербное воскресенье станет для нас началом глубокой внутренней работы: покаяния, примирения и внимательной молитвы в преддверии Страстной седмицы. Аминь.",
                        "Будем просить Господа очистить наши сердца, чтобы встреча с Ним была истинной: не внешней только, но наполненной послушанием, смирением и любовью. Аминь.",
                    ],
                    salt=2,
                )
            else:
                conclusion = pick(
                    [
                        f"Примем праздник «{topic}» как призыв к обновлению сердца: сохраним молитву, благодарность и мир с ближними, чтобы благодать праздника приносила плод в каждом дне. Аминь.",
                        f"Пусть память о празднике «{topic}» укрепит нас в вере и даст силы жить по Евангелию не только в храме, но и дома, в трудах и отношениях с людьми. Аминь.",
                        f"Будем просить Господа, чтобы через праздник «{topic}» Он даровал нам трезвение, духовную собранность и решимость идти путем Христовым. Аминь.",
                    ],
                    salt=2,
                )
        elif self._is_saint_topic(topic_low):
            conclusion = pick(
                [
                    f"По молитвам {topic} будем просить Господа о крепости веры, смирении сердца и мужестве жить по заповедям Христовым в каждом дне. Аминь.",
                    f"Пусть пример {topic} научит нас постоянству в молитве, терпению в испытаниях и деятельной любви к ближнему. С этой решимостью продолжим путь ко Христу. Аминь.",
                    f"Не ограничимся лишь словами о святости: начнем подражать примеру {topic} в покаянии, милосердии и верности церковной жизни. Да укрепит нас в этом Господь. Аминь.",
                ],
                salt=2,
            )
        elif is_sin_topic:
            conclusion = pick(
                [
                    f"Не оправдывайте {sin_name_nominative} и не откладывайте покаяние: время исправления - сегодня. Попросим у Господа решимости отсечь страсть, очистить совесть и начать новую жизнь во Христе. Аминь.",
                    f"Будем строги к собственной душе и милостивы к ближним: отвергнем привычку {sin_name_nominative}, принесем честное покаяние и укрепимся в делах света. Да поможет нам Господь. Аминь.",
                    f"Пусть это слово не останется рассуждением: каждый увидит, где действует {sin_name_nominative}, исповедует это без самооправдания и начнет путь исправления с молитвой и трезвением. Аминь.",
                ],
                salt=2,
            )
        else:
            conclusion = pick(
                [
                    "Будем просить у Господа трезвения ума, смирения сердца и решимости жить по Евангелию в каждом дне. Пусть наша вера станет светом для ближних и источником мира в доме. Аминь.",
                    "Не отложим духовное исправление на потом: начнем сегодня с молитвы, примирения и доброго дела ради Христа. Да укрепит нас Господь на пути спасения и дарует радость о Нем. Аминь.",
                    "Пусть в наших семьях умножаются мир, прощение и милосердие, а сердце каждого будет открыто для благодати Божией. С надеждой на Христа и пойдем дальше по пути веры. Аминь.",
                ],
                salt=2,
            )
        if rng.random() < 0.5:
            conclusion += " " + pick(
                [
                    "Да подаст нам Господь силы хранить это решение в каждом дне.",
                    "Пусть Пресвятая Богородица покрывает нас Своим заступничеством.",
                    "С благодарением Богу продолжим путь христианской жизни в мире и взаимной поддержке.",
                ],
                salt=15,
            )
        if self._is_lazarus_topic(req):
            conclusion_extension_pool = [
                "Вступая в дни Страстной седмицы, сохраним трезвение, молитву и милосердие к ближним.",
                "Не будем откладывать исправление на потом, но начнем сегодня с малого шага покаяния и примирения.",
                "Пусть Господь укрепит нас, чтобы память о Лазаре стала в нас живой надеждой и духовной решимостью.",
                "Доверим Богу свои скорби и немощи, зная, что Его милость сильнее нашей слабости.",
            ]
        elif self._is_prodigal_topic(req):
            conclusion_extension_pool = [
                "Пусть каждый наш день будет шагом домой, к Богу, через верность молитве и честность перед совестью.",
                "Не будем судить других с холодностью старшего брата, но научимся милости и состраданию.",
                "Если мы упали, не останемся в отчаянии, а снова поднимемся и пойдем к Отцу с покаянным сердцем.",
                "Да поможет нам Господь хранить благодарность за Его долготерпение и щедрую любовь к человеку.",
            ]
        elif event_profile is not None:
            conclusion_extension_pool = [
                "Пусть это решение не останется только словами, но станет живым правилом нашей повседневной жизни.",
                "Попросим у Господа помощи хранить верность этому слову в семье, в труде и в отношениях с людьми.",
                "Да укрепит нас Бог, чтобы мы не теряли духовной собранности и надежды среди трудностей дня.",
                "Будем помнить, что благодать действует там, где человек с терпением и смирением начинает исполнять Евангелие.",
            ]
        elif self._is_resurrection_topic(req):
            conclusion_extension_pool = [
                "Сохраним пасхальный свет не только в храме, но и в доме, в семье, в наших словах и поступках.",
                "Будем свидетелями Воскресшего Господа через терпение, кротость и деятельную любовь к людям.",
                "Пусть радость о Христе укрепляет нас в скорбях и учит благодарить Бога за каждый прожитый день.",
                "Да поможет нам Господь хранить верность Евангелию, чтобы пасхальная радость была подлинной и глубокой.",
            ]
        elif self._is_feast_topic(topic_low):
            conclusion_extension_pool = [
                "Принесем плод праздника в наши будни: больше молитвы, больше благодарения и больше милосердия.",
                "Постараемся сохранить в сердце услышанное слово и воплотить его в конкретных делах любви.",
                "Пусть благодать церковного торжества укрепит нас в вере, трезвении и мире с ближними.",
                "Да дарует Господь каждому из нас духовную собранность и верность на пути спасения.",
            ]
        elif self._is_saint_topic(topic_low):
            conclusion_extension_pool = [
                "Попросим Господа, чтобы по молитвам святого Он укрепил нас в постоянстве и чистоте сердца.",
                "Пусть пример угодника Божия вдохновит нас на верность молитве и терпение в испытаниях.",
                "Не ограничимся только добрыми словами, но начнем подражать святому в делах милосердия и правды.",
                "Да поможет нам Бог идти путем покаяния, чтобы и наша жизнь стала свидетельством Его благодати.",
            ]
        elif is_sin_topic:
            conclusion_extension_pool = [
                f"Будем ежедневно проверять совесть и отсекать проявления {sin_name_genitive} в самом начале, пока страсть не укоренилась.",
                f"Не ограничимся эмоциональным сожалением, но подтвердим покаяние делами: {sin_practice}.",
                "Пусть страх Божий и память о Евангелии охраняют нас от возвращения к прежним падениям.",
                "Да укрепит нас Господь в трезвении и воздержании, чтобы свобода от страсти стала реальностью нашей жизни.",
            ]
        else:
            conclusion_extension_pool = [
                "Сделаем сегодняшний выбор практическим: примиримся, поблагодарим Бога и поддержим нуждающихся рядом.",
                "Пусть это слово сопровождает нас в течение недели и помогает хранить мирное устроение души.",
                "Попросим у Господа мудрости в решениях, чистоты в мыслях и терпения в отношении к ближним.",
                "Да укрепит нас Бог, чтобы вера была для нас не только словом, но и образом жизни.",
            ]
        conclusion += " " + " ".join(pick_many(conclusion_extension_pool, 3))
        if "амин" not in conclusion.lower():
            conclusion = conclusion.rstrip() + " Аминь."
        conclusion = self._dedupe_sentences(conclusion)
        intro = self._dedupe_sentences(intro)
        intro = self._apply_orthodox_casing(intro)
        main = self._apply_orthodox_casing(main)
        conclusion = self._apply_orthodox_casing(conclusion)

        title = self._compose_title(req)
        sermon = (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{conclusion}"
        )
        sermon = self._enforce_topic_lock(sermon, req)
        sermon = self._tighten_main_repetition(sermon, req)
        sermon = self._ensure_paschal_conclusion(sermon, req)
        sermon = self._ensure_amen_last(sermon, req)
        return sermon

    def analyze(self, req: AnalyzeRequest) -> AnalyzeResponse:
        clean_text = self.preprocessor.normalize(req.text)[: self.settings.max_input_chars]
        req = AnalyzeRequest(text=clean_text, question=req.question, top_k_sources=req.top_k_sources)

        retrieval_query = f"{req.question or ''} {req.text}"
        citations = self.retrieval.search(retrieval_query, top_k=req.top_k_sources)
        themes = self.preprocessor.extract_themes(clean_text)

        prompt = self._build_analysis_prompt(req, citations)
        generated = self.generator.generate(
            prompt=prompt,
            max_new_tokens=320,
            temperature=0.65,
            top_p=0.9,
            repetition_penalty=1.1,
        )

        analysis_text = generated.text.strip()
        if not analysis_text:
            analysis_text = (
                "В анализируемом отрывке можно выделить темы покаяния, веры и практики духовной жизни. "
                "Рекомендуется сопоставить текст с толкованиями святых отцов и литургическим контекстом."
            )

        return AnalyzeResponse(
            analysis=analysis_text,
            key_themes=themes,
            citations=citations,
            disclaimer=DISCLAIMER,
        )

    def generate_sermon(self, req: GenerateRequest) -> GenerateResponse:
        topic = self.preprocessor.normalize(req.topic or "")
        user_prompt = self.preprocessor.normalize(req.prompt or "")
        bible_text = self.preprocessor.normalize(req.bible_text or "")
        candidate_pool: List[str] = []

        retrieval_query = " ".join(
            part for part in [user_prompt, topic, bible_text, req.occasion or ""] if part
        )
        retrieval_top_k = min(36, max(req.top_k_sources * 4, req.top_k_sources + 10))
        citations_raw = self.retrieval.search(retrieval_query, top_k=retrieval_top_k)
        citations_diverse = self._diversify_citations(citations_raw, req)
        citations = self._select_citation_window(citations_diverse, req)

        if user_prompt:
            prompt = self._build_user_prompt_mode(req, citations)
        else:
            prompt = self._build_sermon_prompt(req, citations)
        generated = self.generator.generate(
            prompt=prompt,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty,
        )
        cleaned = self._cleanup_sermon_text(generated.text)
        sermon = self._format_three_part_sermon(cleaned, req, citations)
        candidate_pool.append(sermon)

        needs_retry = (not self._is_structured_sermon(sermon)) or self._is_noisy_sermon(
            sermon, require_structure_markers=False
        )
        if needs_retry:
            retry_prompt = (
                prompt
                + "\n\nВажно: проповедь должна быть цельной, богословски связной и содержать три полноценных "
                "раздела: вступление, основная часть, заключение. Избегай разговорного потока, списков, "
                "методических инструкций и служебных пометок."
            )
            retry = self.generator.generate(
                prompt=retry_prompt,
                max_new_tokens=max(req.max_new_tokens, 680),
                temperature=max(0.62, req.temperature - 0.12),
                top_p=min(0.92, req.top_p),
                repetition_penalty=max(1.12, req.repetition_penalty),
            )
            retry_cleaned = self._cleanup_sermon_text(retry.text)
            sermon = self._format_three_part_sermon(retry_cleaned, req, citations)
            candidate_pool.append(sermon)

        # Вторая попытка генерации моделью перед fallback, чтобы реже скатываться в шаблон.
        if self._is_noisy_sermon(sermon, require_structure_markers=False):
            retry2_prompt = (
                prompt
                + "\n\nПиши как священник на амвоне: без мета-инструкций, без учебных указаний, "
                "без обращений к пользователю, только цельная проповедь."
            )
            retry2 = self.generator.generate(
                prompt=retry2_prompt,
                max_new_tokens=max(req.max_new_tokens, 760),
                temperature=min(0.95, max(0.74, req.temperature + 0.04)),
                top_p=min(0.96, max(0.9, req.top_p)),
                repetition_penalty=max(1.15, req.repetition_penalty),
            )
            retry2_cleaned = self._cleanup_sermon_text(retry2.text)
            sermon = self._format_three_part_sermon(retry2_cleaned, req, citations)
            candidate_pool.append(sermon)

        if not self._topic_is_covered(sermon, req):
            topic = self._extract_topic(req)
            retry_topic_prompt = (
                prompt
                + f"\n\nКритически важно: проповедь должна быть именно о теме «{topic}»."
                " Раскрой тему по существу в основной части, а не только упомяни в заголовке."
            )
            retry_topic = self.generator.generate(
                prompt=retry_topic_prompt,
                max_new_tokens=max(req.max_new_tokens, 760),
                temperature=min(0.92, max(0.72, req.temperature)),
                top_p=min(0.95, max(0.9, req.top_p)),
                repetition_penalty=max(1.14, req.repetition_penalty),
            )
            retry_topic_cleaned = self._cleanup_sermon_text(retry_topic.text)
            sermon = self._format_three_part_sermon(retry_topic_cleaned, req, citations)
            candidate_pool.append(sermon)

        picked = self._pick_best_candidate(candidate_pool, req)
        if picked:
            sermon = picked

        # Финальная защита от мусора: если текст токсично шумный, уходим в safe-режим.
        if self._is_noisy_sermon(sermon, require_structure_markers=True):
            sermon = self._compose_safe_sermon(req, citations)
            candidate_pool.append(sermon)
        else:
            low = sermon.lower()
            has_sections = all(x in low for x in ["вступление.", "основная часть.", "заключение."])
            if not has_sections:
                sermon = self._format_three_part_sermon(sermon, req, citations)
                if self._is_noisy_sermon(sermon, require_structure_markers=True):
                    sermon = self._compose_safe_sermon(req, citations)
                    candidate_pool.append(sermon)
        if not self._main_is_substantial(sermon):
            sermon = self._compose_safe_sermon(req, citations)
            candidate_pool.append(sermon)
        if not self._topic_is_covered(sermon, req):
            sermon = self._compose_safe_sermon(req, citations)
            candidate_pool.append(sermon)
        sermon = self._ensure_quote_paragraphs(sermon, req, citations)
        sermon = self._enforce_topic_lock(sermon, req)
        sermon = self._tighten_main_repetition(sermon, req)
        sermon = self._ensure_quote_paragraphs(sermon, req, citations)
        sermon = self._enforce_topic_lock(sermon, req)
        if not self._main_is_substantial(sermon):
            sermon = self._compose_safe_sermon(req, citations)
            sermon = self._ensure_quote_paragraphs(sermon, req, citations)
            sermon = self._enforce_topic_lock(sermon, req)
            sermon = self._tighten_main_repetition(sermon, req)
            candidate_pool.append(sermon)
        if not self._topic_is_covered(sermon, req):
            sermon = self._compose_safe_sermon(req, citations)
            sermon = self._ensure_quote_paragraphs(sermon, req, citations)
            sermon = self._enforce_topic_lock(sermon, req)
            sermon = self._tighten_main_repetition(sermon, req)
            candidate_pool.append(sermon)

        final_pick = self._pick_best_candidate(candidate_pool + [sermon], req)
        if final_pick:
            sermon = final_pick
            sermon = self._ensure_quote_paragraphs(sermon, req, citations)
            sermon = self._enforce_topic_lock(sermon, req)
            sermon = self._tighten_main_repetition(sermon, req)

        if self._is_too_similar_to_recent(sermon):
            alt_pool = [sermon]
            for _ in range(2):
                alt = self._compose_safe_sermon(req, citations)
                alt = self._ensure_quote_paragraphs(alt, req, citations)
                alt = self._enforce_topic_lock(alt, req)
                alt = self._tighten_main_repetition(alt, req)
                alt_pool.append(alt)
            diverse_pick = self._pick_best_candidate(alt_pool, req)
            if diverse_pick:
                sermon = diverse_pick

        sermon = self._ensure_quote_paragraphs(sermon, req, citations)
        sermon = self._enforce_topic_lock(sermon, req)
        sermon = self._tighten_main_repetition(sermon, req)
        sermon = self._apply_orthodox_casing(sermon)
        sermon = self._ensure_paschal_conclusion(sermon, req)
        sermon = self._ensure_amen_last(sermon, req)
        self._remember_sermon(sermon)
        quality = self._build_quality_metrics(sermon, req)

        outline = self._build_outline(sermon)
        return GenerateResponse(
            sermon=sermon,
            outline=outline,
            citations=citations,
            model_name=generated.model_name,
            quality=quality,
            disclaimer=DISCLAIMER,
        )

    def health_flags(self) -> Tuple[bool, bool]:
        return self.generator.loaded, self.generator.adapter_loaded
