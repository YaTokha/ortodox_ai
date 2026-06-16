import re
from typing import List


class TextPreprocessor:
    """Небольшой модуль очистки текста для запросов и корпуса."""

    _space_regex = re.compile(r"\s+")

    _abbr_placeholder = "§"

    def normalize(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = self._space_regex.sub(" ", text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        text = self.normalize(text)
        if not text:
            return []
        protected = text

        # Защищаем сокращения в библейских ссылках, чтобы не обрывать "Мих. 6:8" на "Мих."
        def _protect_scripture_ref(match: re.Match) -> str:
            book = match.group(1)
            rest = match.group(2)
            return f"{book}{self._abbr_placeholder} {rest}"

        protected = re.sub(
            r"\b((?:[1-3]\s*)?[А-ЯЁ][а-яё]{1,10})\.\s*(\d{1,3}:\d{1,3}(?:-\d{1,3})?)",
            _protect_scripture_ref,
            protected,
        )

        # Защищаем базовые церковные сокращения в именованиях.
        protected = re.sub(
            r"\b(Свт|Прп|Блж|Свящ|Прот|Митр|Патр)\.\s+",
            lambda m: f"{m.group(1)}{self._abbr_placeholder} ",
            protected,
        )

        parts = re.split(r"(?<=[.!?])\s+", protected)
        restored = [p.replace(self._abbr_placeholder, ".").strip() for p in parts if p.strip()]
        return restored

    def extract_themes(self, text: str) -> List[str]:
        text_low = self.normalize(text).lower()
        themes = []
        dictionary = {
            "покаяние": ["покая", "грех", "исповед"],
            "любовь": ["любов", "ближн"],
            "смирение": ["смир", "гордын"],
            "молитва": ["молит", "молитесь", "пост"],
            "милосердие": ["милосер", "помощ", "сострадан"],
            "вера и надежда": ["вера", "надежд", "упован"],
        }
        for theme, markers in dictionary.items():
            if any(marker in text_low for marker in markers):
                themes.append(theme)
        if not themes:
            themes = ["духовная жизнь", "толкование текста", "практика христианской жизни"]
        return themes
