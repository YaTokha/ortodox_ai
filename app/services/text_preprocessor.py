import re
from typing import List


class TextPreprocessor:
    """Небольшой модуль очистки текста для запросов и корпуса."""

    _space_regex = re.compile(r"\s+")

    def normalize(self, text: str) -> str:
        text = text.replace("\u00a0", " ")
        text = self._space_regex.sub(" ", text)
        return text.strip()

    def split_into_sentences(self, text: str) -> List[str]:
        text = self.normalize(text)
        if not text:
            return []
        # Простое деление по знакам конца предложения.
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

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
