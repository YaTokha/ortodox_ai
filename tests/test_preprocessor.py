from app.services.text_preprocessor import TextPreprocessor


def test_normalize_and_sentence_split() -> None:
    p = TextPreprocessor()
    text = "  Это   первое предложение.  Это второе!   "
    assert p.normalize(text) == "Это первое предложение. Это второе!"
    assert p.split_into_sentences(text) == ["Это первое предложение.", "Это второе!"]


def test_extract_themes() -> None:
    p = TextPreprocessor()
    themes = p.extract_themes("Через молитву и смирение человек получает надежду.")
    assert "молитва" in themes
    assert "смирение" in themes
