import re
import hashlib
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
            "8) Не делай списки и перечисления через «;», пиши плавными связными фразами.\n"
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
            "Не используй перечисления через «;», служебные пометки и телеграфный стиль.\n"
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
        return f"Проповедь на тему: «{topic.capitalize()}»"

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
        instruction_markers = [
            "не нужно делать ссылок",
            "цитируй",
            "без ссылок",
            "отсутствие ссылок",
            "структура:",
            "включение:",
            "запрос пользователя",
            "проповеди:",
        ]
        if sum(1 for m in instruction_markers if m in low) >= 1:
            return True
        if norm.count(":") >= 6:
            return True
        low_spaced = f" {low} "
        if low_spaced.count(" если вы хотите ") >= 2:
            return True
        if low_spaced.count(" вы можете ") >= 2:
            return True
        if low_spaced.count(" я ") >= 4 or low_spaced.count(" меня ") >= 2:
            return True
        first_singular = len(re.findall(r"\b(я|мне|меня|мой|моя|моё|мое)\b", low))
        second_singular = len(re.findall(r"\b(ты|тебе|тебя|твой|твоя|твое|твоё)\b", low))
        if first_singular >= 3 or second_singular >= 2:
            return True
        if "если ты" in low or "не молись" in low:
            return True
        if low_spaced.count(" то есть ") >= 4:
            return True
        if sum(1 for w in ["приведи", "приводите", "цитируй", "вставляй"] if w in low) >= 1:
            return True
        if any(m in low for m in ["краткая, содержательная, логичная", "без излишней эмоциональности"]):
            return True
        if any(m in low for m in ["вот тебе пример", "я тут родилась", "ко мне, исповедовать"]):
            return True
        if norm.count("?") >= 4:
            return True
        # Слишком плотные повторы слов -> низкое качество.
        if words:
            uniq_ratio = len({w.lower() for w in words}) / len(words)
            if uniq_ratio < 0.45:
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
        return (
            f"{title}\n\n"
            f"Вступление.\n{intro}\n\n"
            f"Основная часть.\n{main}\n\n"
            f"Заключение.\n{concl}"
        )

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
        if self._is_section_poor(intro, min_words=22):
            return False
        if self._is_section_poor(main, min_words=70):
            return False
        if self._is_section_poor(concl, min_words=18):
            return False
        if sum(ch in ".!?" for ch in main) < 3:
            return False
        if text.count(";") >= 12:
            return False
        if text.count(":") >= 16:
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
        audience = self.preprocessor.normalize(req.audience or "прихода")
        audience_low = audience.lower()

        seed_src = f"{topic}|{bible_ref}|{occasion}|{audience}"
        seed = int(hashlib.sha256(seed_src.encode("utf-8")).hexdigest()[:8], 16)

        def pick(options: List[str], salt: int = 0) -> str:
            idx = (seed + salt) % len(options)
            return options[idx]

        cited_authors = [c.author for c in citations if c.author]
        cited_authors = [x for i, x in enumerate(cited_authors) if x and x not in cited_authors[:i]][:2]
        if cited_authors:
            fathers_line = (
                f"Опыт Церкви, раскрытый в слове {cited_authors[0]}"
                + (f" и {cited_authors[1]}" if len(cited_authors) > 1 else "")
                + ", напоминает: духовная жизнь требует не теории, а постоянного внутреннего делания."
            )
        else:
            fathers_line = (
                "Святоотеческая традиция напоминает: духовная жизнь требует не внешней формы, "
                "а постоянного внутреннего делания перед Богом."
            )

        topic_low = topic.lower()
        if any(w in topic_low for w in ["воскрес", "пасх", "побед", "жизнь вечн"]):
            doctrinal = pick(
                [
                    "Воскресение Христово открывает нам не просто память о событии, а новую реальность: смерть уже не имеет последнего слова, потому что Христос победил ад и даровал человеку путь к вечной жизни.",
                    "Пасхальная весть говорит каждому сердцу: Бог не оставил человека в плену тьмы, но Сам вошел в глубину человеческой боли, чтобы вывести нас к свету воскресения и надежды.",
                    "Тайна Воскресения учит нас, что Божия любовь сильнее греха, страха и отчаяния, а значит, даже в самых тяжелых обстоятельствах христианин может жить надеждой и мужеством веры.",
                ]
            )
            practice = pick(
                [
                    "Будем хранить пасхальную радость не только в словах, но и в делах: примиряться с ближними, поддерживать тех, кто в скорби, и благодарить Бога за каждый день как за дар новой жизни.",
                    "Пусть вера в Воскресшего Христа выражается в конкретной заботе о семье, в терпении к немощам друг друга и в милосердии к тем, кто нуждается в нашем времени и участии.",
                    "Если Христос воскрес, значит, и наша повседневность может быть преображена: оставим уныние, будем внимательны к молитве и станем носителями мира там, где прежде было раздражение и холодность.",
                ],
                salt=1,
            )
        elif any(w in topic_low for w in ["покая", "исповед", "грех", "осужд"]):
            doctrinal = pick(
                [
                    "Истинное покаяние начинается там, где человек перестает оправдывать себя и в смирении открывает сердце Богу. Господь принимает кающегося не для осуждения, а для исцеления и обновления жизни.",
                    "Покаяние — это не разовое чувство, а путь возвращения к Богу: через трезвение ума, честность перед совестью и готовность менять свои поступки, а не только слова.",
                    "Когда человек кается, благодать Божия постепенно исцеляет внутренние раны и учит жить по-новому: с миром, терпением и любовью к ближнему.",
                ]
            )
            practice = pick(
                [
                    "Начнем с малого: попросим прощения у тех, кого обидели, прекратим привычку осуждать и ежедневно принесем Богу искреннюю молитву покаяния. Эти шаги просты, но именно в них рождается новая жизнь.",
                    "Будем чаще прибегать к исповеди, внимать своим словам и мыслям, беречь сердце от жесткости. Так постепенно исчезает внутренняя горечь, и человек становится способным к милосердию.",
                    "Путь покаяния требует верности в малом: хранить совесть, не откладывать исправление и делать добро ради Христа. Так в душе утверждается мир, который невозможно заменить внешним успехом.",
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

        intro_body = pick(
            [
                f"Сегодня, в день {occasion}, обратимся к размышлению о {topic}. Эта тема касается каждого из нас, потому что именно в обычных обстоятельствах раскрывается подлинная глубина веры.",
                f"В день {occasion} Церковь вновь напоминает нам о {topic}. Для {audience} это не отвлеченное рассуждение, а живой вопрос духовного пути и ответственности перед Богом.",
                f"Обращаясь к теме {topic}, будем помнить: Господь ждет от нас не красивых слов, а реального движения сердца к Нему. Именно так человек духовно взрослеет и укрепляется в истине.",
            ]
        )
        intro = "Во имя Отца, и Сына, и Святого Духа!\nДорогие братья и сестры!\n" + intro_body

        first_main = (
            f"Евангельский фрагмент {bible_ref} направляет нас к живой вере и упованию на Господа. {doctrinal}"
            if bible_ref
            else doctrinal
        )
        second_main = fathers_line
        third_main = practice + " Если же мы падаем, не будем отчаиваться: Господь поднимает кающегося и укрепляет его на новом пути."
        if any(x in audience_low for x in ["молод", "студент", "подрост"]):
            fourth_main = (
                "Особенно важно сказать об этом молодежи: духовная жизнь строится не мгновенно, а через верность в малом. "
                "Когда человек учится хранить чистоту мысли, уважение к родителям, ответственность в учебе и труде, "
                "тогда вера становится прочным основанием всей его жизни."
            )
        elif any(x in audience_low for x in ["сем", "родител", "супруг"]):
            fourth_main = (
                "Для семейной жизни эта тема имеет особую силу: дом становится по-настоящему христианским там, "
                "где есть совместная молитва, взаимное прощение и готовность нести тяготы друг друга. "
                "Через такую верность в повседневности Господь дарует семье мир и единство."
            )
        else:
            fourth_main = (
                "Пусть каждое наше решение проходит через вопрос совести: ведет ли оно к миру, любви и истине Христовой. "
                "Тогда даже обычные обязанности дня становятся частью духовного пути, а сердце постепенно укрепляется в надежде на Бога."
            )

        main = f"{first_main}\n\n{second_main}\n\n{third_main}\n\n{fourth_main}"

        conclusion = pick(
            [
                "Будем просить у Господа трезвения ума, смирения сердца и решимости жить по Евангелию в каждом дне. Пусть наша вера станет светом для ближних и источником мира в доме. Аминь.",
                "Не отложим духовное исправление на потом: начнем сегодня с молитвы, примирения и доброго дела ради Христа. Да укрепит нас Господь на пути спасения и дарует радость о Нем. Аминь.",
                "Пусть в наших семьях умножаются мир, прощение и милосердие, а сердце каждого будет открыто для благодати Божией. С надеждой на Христа и пойдем дальше по пути веры. Аминь.",
            ],
            salt=2,
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

        if not self._is_structured_sermon(sermon):
            sermon = self._compose_safe_sermon(req, citations)

        if len(sermon) < 1300:
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
