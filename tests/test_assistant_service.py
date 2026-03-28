import os

os.environ["DISABLE_MODEL"] = "true"

from app.config import get_settings
from app.schemas import GenerateRequest
from app.services.assistant_service import OrthodoxAssistantService
from app.services.generation import GenerationResult


def test_generate_sermon_filters_html_noise_and_returns_coherent_text() -> None:
    noisy_text = """
Иоанн Златоуст. О том, как Иисус Христос впервые начал учить и проповедовать через веру.
<!--
 /* Style Definitions */
 p.MsoNormal, li.MsoNormal, div.MsoNormal
    {mso-style-unhide:no; font-family:"Times New Roman","serif";}
@page Section1
    {mso-paper-source:0;}
-->
<br />
Апостол Павел, будучи в Иерусалиме, беседовал со своими учениками...
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=noisy_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]

    res = service.generate_sermon(
        GenerateRequest(prompt="сгенерируй проповедь о покаянии", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert "style definitions" not in low
    assert "mso-" not in low
    assert "<!--" not in low
    assert "<br" not in low
    assert "план:" not in low
    assert "источники:" not in low
    assert "fallback-режим" not in low
    assert "покаяни" in low
    assert low.startswith("проповедь:")
    assert "во имя отца, и сына, и святого духа!" in low
    assert "дорогие братья и сестры!" in low
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "амин" in low


def test_generate_sermon_filters_citation_dump_lines() -> None:
    noisy_text = """
- commentary; Блж. Феофилакт Болгарский; Толкование Евангелия; Источник: https://royallib.com/get/txt/feofilakt_blg/tolkovanie_na_evangelie_ot_marka.zip: Не слушайте тех, кто говорит...
- commentary; Блж. Феофилакт Болгарский; Толкование Евангелия; Источник: https://royallib.com/get/txt/feofilakt_blg/tolkovanie_na_evangelie_ot_marka.zip: Он говорил вам: "Веруете ли вы?"
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=noisy_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]

    res = service.generate_sermon(
        GenerateRequest(prompt="сгенерируй проповедь о покаянии", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert "commentary;" not in low
    assert "источник:" not in low
    assert "https://" not in low
    assert low.startswith("проповедь:")
    assert "во имя отца, и сына, и святого духа!" in low
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "покаяни" in low
    assert "амин" in low


def test_generate_sermon_removes_direct_quotes_and_keeps_three_parts() -> None:
    quoted_text = (
        "Вступление. Как сказано: \"Покайтесь, ибо приблизилось Царство Небесное\". "
        "Основная часть. Это важное слово. "
        "Заключение. Аминь."
    )
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=quoted_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="сгенерируй проповедь о покаянии", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert low.startswith("проповедь:")
    assert "во имя отца, и сына, и святого духа!" in low
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert '"' not in res.sermon
    assert "как сказано" not in low


def test_generate_sermon_filters_rule_dump_and_metadata() -> None:
    noisy_text = """
да, нет, конечно! Иисус был не пророк, а наставник.
Правило 1:
Исповедовать Евангелие необходимо всем христианам.
Правило 2:
Библия учит нас, что любовь – это естественное состояние человека.
16243862    royallib.ru    2018-08-01 19:42:00    Отдыхая с пользой
Интересное на LiveJ
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=noisy_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="сгенерируй проповедь о покаянии", top_k_sources=2)
    )
    low = res.sermon.lower()
    assert low.startswith("проповедь:")
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "правило 1:" not in low
    assert "royallib" not in low
    assert "livej" not in low
