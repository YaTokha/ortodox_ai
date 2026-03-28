import re
from html import unescape
from typing import List, Tuple

from app.config import Settings
from app.schemas import AnalyzeRequest, AnalyzeResponse, Citation, GenerateRequest, GenerateResponse
from app.services.generation import SermonGenerator
from app.services.retrieval import CorpusRetrievalService
from app.services.text_preprocessor import TextPreprocessor

DISCLAIMER = (
    "Материал сгенерирован ИИ и предназначен как черновик для подготовки. "
    "Перед использованием требуется богословская проверка священнослужителем."
)


class OrthodoxAssistantService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.preprocessor = TextPreprocessor()
        self.retrieval = CorpusRetrievalService(settings.corpus_abspath())
        self.generator = SermonGenerator(settings)

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
        return (
            "Напиши цельную православную проповедь на русском языке.\n"
            "Требования:\n"
            "1) Верни только готовый связный текст проповеди.\n"
            "2) Структура: вступление, основная часть, заключение (цельными абзацами).\n"
            "3) Тон пастырский, спокойный, назидательный.\n"
            "4) Начало: «Во имя Отца, и Сына, и Святого Духа!» и обращение к пастве.\n"
            "5) Основа: евангельский смысл и святоотеческая традиция в пересказе, без прямых цитат.\n"
            "6) Завершение: практический призыв к покаянию/добрым делам и финал «Аминь.»\n"
            "7) Не вставляй ссылки, служебные метки, названия сайтов и технические пометки.\n"
            f"Тема: {req.topic}\n"
            f"Повод/праздник: {req.occasion or 'обычное богослужение'}\n"
            f"Аудитория: {req.audience or 'приход'}\n"
            f"Стиль: {req.style}\n"
            f"Библейский фрагмент: {req.bible_text or 'не указан'}\n"
            "Опора: Священное Писание и святоотеческая православная традиция.\n"
            "Проповедь:"
        )

    def _build_user_prompt_mode(self, user_prompt: str, citations: List[Citation]) -> str:
        return (
            "Ты православный ассистент для подготовки проповедей.\n"
            "Сгенерируй цельную проповедь по запросу пользователя.\n"
            "Структура: вступление, основная часть, заключение (абзацы, не списки).\n"
            "Начало: «Во имя Отца, и Сына, и Святого Духа!» и обращение к пастве.\n"
            "Без прямых цитат; передавай смысл Писания и святых отцов своими словами.\n"
            "Не добавляй ссылки, имена файлов, названия сайтов, метки типа commentary/sermon и технические вставки.\n"
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
        topic = self._extract_topic(req).strip(" .,:;!?").lower()
        if not topic:
            topic = "христианской жизни"
        if topic.startswith(("о ", "об ")):
            return f"Проповедь: «{topic.capitalize()}»"
        return f"Проповедь: «О {topic}»"

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
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{concl}"
        )

    def _is_structured_sermon(self, text: str) -> bool:
        low = (text or "").lower()
        if not low.startswith("проповедь:"):
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
        return True

    def _extract_topic(self, req: GenerateRequest) -> str:
        topic = self.preprocessor.normalize(req.topic or "")
        if topic:
            return topic

        prompt = self.preprocessor.normalize(req.prompt or "")
        if not prompt:
            return "христианская жизнь"

        # Убираем типичные императивные префиксы пользовательского промта.
        patterns = [
            r"^(сгенерируй|составь|подготовь|напиши|создай)\s+",
            r"^(кратк\w+|цельн\w+)\s+",
            r"^(православн\w+)\s+",
            r"^проповед\w*\s+(о|про|на тему)\s+",
        ]
        topic_guess = prompt.lower()
        for p in patterns:
            topic_guess = re.sub(p, "", topic_guess, flags=re.IGNORECASE)
        topic_guess = topic_guess.strip(" .,:;!-?")

        if len(topic_guess) < 3:
            return "христианская жизнь"
        return topic_guess

    def _compose_safe_sermon(self, req: GenerateRequest, citations: List[Citation]) -> str:
        topic = self._extract_topic(req)
        bible_ref = self.preprocessor.normalize(req.bible_text or "")
        occasion = self.preprocessor.normalize(req.occasion or "обычного богослужения")

        source_hint = (
            " Святоотеческая традиция напоминает нам о внутреннем покаянии, трезвении и верности Богу."
            if citations
            else ""
        )

        topic_low = topic.lower()
        if any(w in topic_low for w in ["покая", "исповед", "грех"]):
            doctrinal = (
                "Покаяние начинается с честного взгляда на свою совесть и продолжается живым обращением к Богу. "
                "Господь принимает кающегося не для осуждения, а для исцеления и обновления жизни."
            )
            practice = (
                "Будем внимательны к молитве, чаще прибегать к исповеди, хранить сердце от осуждения и учиться прощать. "
                "Так постепенно в нас восстанавливается мир, и мы становимся мягче и милосерднее к ближним."
            )
        else:
            doctrinal = (
                "Евангелие призывает нас не к внешнему благочестию, а к внутреннему преображению сердца. "
                "Когда человек доверяет Богу, он получает силы идти путём веры даже среди скорбей и сомнений."
            )
            practice = (
                "Поэтому будем хранить молитву, избегать осуждения, внимать своей совести и чаще обращаться к Таинствам Церкви. "
                "Каждое доброе дело, совершенное ради Христа, укрепляет душу и делает нас способными нести мир ближним."
            )

        intro = (
            "Во имя Отца, и Сына, и Святого Духа!\n"
            "Дорогие братья и сестры!\n"
            f"Сегодня, в день {occasion}, обратимся к размышлению о {topic}. "
            "Эта тема касается каждого из нас, потому что именно в ежедневной жизни проверяется глубина нашей веры, "
            "терпения и любви к ближним."
        )
        main = (
            f"{'Евангельский фрагмент ' + bible_ref + ' направляет нас к живой вере и упованию на Господа. ' if bible_ref else ''}"
            + doctrinal
            + source_hint
            + " "
            + practice
            + " "
            + "Если же мы падаем, не будем отчаиваться: Господь поднимает кающегося и укрепляет его на новом пути."
        )
        conclusion = (
            "Пусть Господь дарует нам трезвение ума, смирение сердца и живую любовь к людям, "
            "чтобы евангельская истина стала для нас не только знанием, но и образом жизни. Аминь."
        )

        title = self._compose_title(req)
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{conclusion}"
        )

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

        retrieval_query = " ".join(
            part for part in [user_prompt, topic, bible_text, req.occasion or ""] if part
        )
        citations = self.retrieval.search(retrieval_query, top_k=req.top_k_sources)

        if user_prompt:
            prompt = self._build_user_prompt_mode(user_prompt, citations)
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

        if not self._is_structured_sermon(sermon):
            sermon = self._compose_safe_sermon(req, citations)

        if len(sermon) < 120:
            sermon = self._compose_safe_sermon(req, citations)

        outline = self._build_outline(sermon)
        return GenerateResponse(
            sermon=sermon,
            outline=outline,
            citations=citations,
            model_name=generated.model_name,
            disclaimer=DISCLAIMER,
        )

    def health_flags(self) -> Tuple[bool, bool]:
        return self.generator.loaded, self.generator.adapter_loaded
