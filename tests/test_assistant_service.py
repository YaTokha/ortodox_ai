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
    assert low.startswith("проповедь:") or low.startswith("проповедь на тему:")
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
    assert low.startswith("проповедь:") or low.startswith("проповедь на тему:")
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
    assert low.startswith("проповедь:") or low.startswith("проповедь на тему:")
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
    assert low.startswith("проповедь:") or low.startswith("проповедь на тему:")
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "правило 1:" not in low
    assert "royallib" not in low
    assert "livej" not in low


def test_generate_sermon_keeps_requested_bogoroditsa_topic() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добрые дела. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]

    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о Богородице", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "амин" in low
    assert any(m in low for m in ["богород", "пресвят", "божией матери", "божией матери"])


def test_generate_sermon_resurrection_has_paschal_final() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Воскресение Христово открывает путь к надежде и жизни.
Заключение. Будем хранить веру и благодарить Бога. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о Воскрсении Христовом", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert "воскрес" in low
    assert "христос воскресе" in low
    assert "воистину воскресе" in low


def test_generate_sermon_lazarus_topic_is_specific() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добрые дела. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Сгенерируй проповедь о Лазаревой субботе", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "амин" in low
    assert "лазар" in low
    assert any(marker in low for marker in ["вифан", "четвероднев", "марф", "мария", "страстной седмиц"])
    assert "христос воскресе" not in low
    assert "воистину воскресе" not in low


def test_generate_sermon_contains_scripture_fathers_and_preacher_quotes() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Будем хранить веру и жить по совести.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Сгенерируй проповедь о покаянии", top_k_sources=2)
    )

    low = res.sermon.lower()
    assert ("священное писание говорит:" in low) or ("ветхий завет предупреждает" in low)
    assert ("послание святых апостолов наставляет" in low) or ("священное писание говорит:" in low)
    assert "наставляет:" in low
    assert "проповеднической традиции звучит слово" in low


def test_generate_sermon_trinity_and_entry_are_not_identical() -> None:
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    res_entry = service.generate_sermon(
        GenerateRequest(prompt="Сгенерируй проповедь о Входе Господнем в Иерусалим", top_k_sources=3)
    )
    res_trinity = service.generate_sermon(
        GenerateRequest(
            prompt="Составь проповедь о Дне Святой Троицы и действии благодати Святого Духа в жизни христианина",
            top_k_sources=3,
        )
    )

    low_entry = res_entry.sermon.lower()
    low_trinity = res_trinity.sermon.lower()
    assert res_entry.sermon.strip() != res_trinity.sermon.strip()
    assert any(m in low_entry for m in ["иерусалим", "верб", "вход господ"])
    assert any(m in low_trinity for m in ["троиц", "дух свят", "пятидесят"])


def test_generate_sermon_prodigal_son_is_thematic() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Сгенерируй проповедь опираясь на притчу о блудном сыне", top_k_sources=3)
    )
    low = res.sermon.lower()
    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "блуд" in low
    assert any(m in low for m in ["отец", "покаян", "возвращ"])


def test_generate_sermon_supports_twelve_feasts_and_major_gospel_events() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]

    prompts = [
        "Подготовь проповедь о Рождестве Пресвятой Богородицы",
        "Подготовь проповедь о Воздвижении Креста Господня",
        "Подготовь проповедь о Введении во храм Пресвятой Богородицы",
        "Подготовь проповедь о Рождестве Христовом",
        "Подготовь проповедь о Крещении Господнем",
        "Подготовь проповедь о Сретении Господнем",
        "Подготовь проповедь о Благовещении Пресвятой Богородицы",
        "Подготовь проповедь о Входе Господнем в Иерусалим",
        "Подготовь проповедь о Вознесении Господнем",
        "Подготовь проповедь о Дне Святой Троицы",
        "Подготовь проповедь о Преображении Господнем",
        "Подготовь проповедь об Успении Пресвятой Богородицы",
        "Подготовь проповедь о Тайной Вечере",
        "Подготовь проповедь о Крестных страданиях Христовых",
        "Подготовь проповедь о Нагорной проповеди",
        "Подготовь проповедь о милосердном самарянине",
        "Подготовь проповедь о браке в Кане Галилейской",
        "Подготовь проповедь об исцелении слепорожденного",
    ]

    for prompt in prompts:
        req = GenerateRequest(prompt=prompt, top_k_sources=3)
        res = service.generate_sermon(req)
        low = res.sermon.lower()
        assert "вступление." in low
        assert "основная часть." in low
        assert "заключение." in low
        assert service._topic_is_covered(res.sermon, req)  # noqa: SLF001


def test_generate_sermon_nativity_theotokos_uses_event_branch() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о Рождестве Пресвятой Богородицы", top_k_sources=3)
    )
    low = res.sermon.lower()
    assert "рождеств" in low
    assert "богород" in low
    assert "пришествие спасителя" in low


def test_generate_sermon_last_supper_uses_event_branch() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о Тайной Вечере", top_k_sources=3)
    )
    low = res.sermon.lower()
    assert "тайн" in low
    assert "вечер" in low
    assert "евхарист" in low


def test_generate_sermon_mount_sermon_uses_event_branch() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о Нагорной проповеди", top_k_sources=3)
    )
    low = res.sermon.lower()
    assert "нагорн" in low
    assert "проповед" in low
    assert "блаженств" in low


def test_generate_sermon_sin_topic_is_strict_and_thematic() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    req = GenerateRequest(prompt="Подготовь строгую проповедь о грехах и страстях", top_k_sources=3)
    res = service.generate_sermon(req)
    low = res.sermon.lower()

    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert "амин" in low
    assert service._topic_is_covered(res.sermon, req)  # noqa: SLF001
    assert any(m in low for m in ["грех", "страст", "нераскаян"])
    assert any(m in low for m in ["покаян", "исповед", "исправлен", "воздерж"])
    assert "христос воскресе" not in low


def test_generate_sermon_divination_topic_has_normalized_title() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Проповедь о грехе гадания", top_k_sources=3)
    )

    assert "Проповедь на тему: «Грех гадания»" in res.sermon


def test_generate_sermon_divination_topic_is_specific_and_has_scripture_layers() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    req = GenerateRequest(prompt="Подготовь строгую проповедь о грехе гадания", top_k_sources=3)
    res = service.generate_sermon(req)
    low = res.sermon.lower()

    assert "вступление." in low
    assert "основная часть." in low
    assert "заключение." in low
    assert service._topic_is_covered(res.sermon, req)  # noqa: SLF001
    assert any(m in low for m in ["гадан", "оккульт", "гороскоп", "волшеб"])
    assert "ветхий завет предупреждает" in low
    assert "послание святых апостолов наставляет" in low
    assert "кронштадтский иоанн" not in low


def test_generate_sermon_conclusion_ends_with_amen() -> None:
    generic_text = """
Вступление. Во имя Отца, и Сына, и Святого Духа! Дорогие братья и сестры!
Основная часть. Христианская жизнь требует верности Богу в каждом дне.
Заключение. Будем молиться и творить добро. Аминь. Пусть Господь укрепит нас.
"""
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    def fake_generate(*args, **kwargs):
        return GenerationResult(text=generic_text, model_name="test-model")

    service.generator.generate = fake_generate  # type: ignore[method-assign]
    res = service.generate_sermon(
        GenerateRequest(prompt="Подготовь проповедь о покаянии", top_k_sources=3)
    )
    assert res.sermon.strip().endswith("Аминь.")


def test_author_attribution_is_declined_and_spelling_fixed() -> None:
    get_settings.cache_clear()
    service = OrthodoxAssistantService(get_settings())

    assert service._normalize_author_attribution("Кронштадтский Иоанн") == "Иоанна Кронштадтского"  # noqa: SLF001
    assert service._normalize_author_attribution("Свт. Иоанн Кронштадтский") == "Свт. Иоанна Кронштадтского"  # noqa: SLF001
    assert service._normalize_author_attribution("Свт. Феофан Заторник") == "Свт. Феофана Затворника"  # noqa: SLF001
    assert service._inline_author_attribution("Свт. Феофан Заторник") == "свт. Феофана Затворника"  # noqa: SLF001
