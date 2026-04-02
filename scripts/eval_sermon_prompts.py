#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings
from app.schemas import GenerateRequest
from app.services.assistant_service import OrthodoxAssistantService


def build_prompt_pool() -> List[str]:
    feasts = [
        "Рождестве Пресвятой Богородицы",
        "Воздвижении Креста Господня",
        "Введении во храм Пресвятой Богородицы",
        "Рождестве Христовом",
        "Крещении Господнем",
        "Сретении Господнем",
        "Благовещении Пресвятой Богородицы",
        "Входе Господнем в Иерусалим",
        "Вознесении Господнем",
        "Дне Святой Троицы",
        "Преображении Господнем",
        "Успении Пресвятой Богородицы",
        "Лазаревой субботе",
        "Пасхе и Воскресении Христовом",
    ]
    saints = [
        "святителе Николае Чудотворце",
        "преподобном Сергии Радонежском",
        "святителе Иоанне Златоусте",
        "святителе Василии Великом",
        "святителе Григории Богослове",
        "святителе Спиридоне Тримифунтском",
        "святителе Луке (Войно-Ясенецком)",
        "преподобном Серафиме Саровском",
        "блаженной Матроне Московской",
        "праведном Иоанне Кронштадтском",
        "великомученике Георгии Победоносце",
        "великомученике Димитрии Солунском",
        "святом апостоле Петре",
        "святом апостоле Павле",
        "святом апостоле Иоанне Богослове",
        "равноапостольной Марии Магдалине",
        "равноапостольном князе Владимире",
        "благоверном князе Александре Невском",
        "святителе Тихоне Задонском",
        "преподобном Амвросии Оптинском",
        "преподобном Силуане Афонском",
        "преподобном Паисии Святогорце",
        "преподобном Иоанне Лествичнике",
        "преподобном Макарии Египетском",
        "преподобном Ефреме Сирине",
        "святителе Феофане Затворнике",
        "святителе Игнатии (Брянчанинове)",
        "святителе Афанасии Великом",
        "святителе Кирилле Иерусалимском",
        "святителе Григории Нисском",
        "святителе Филарете Московском",
        "святителе Иннокентии Московском",
        "преподобной Марии Египетской",
        "святой мученице Татиане",
        "святом праведном Иоанне Русском",
        "священномученике Ермогене",
    ]
    themes = [
        "покаянии",
        "исповеди",
        "смирении",
        "молитве",
        "посте",
        "милосердии",
        "любви к ближнему",
        "прощении обид",
        "борьбе с осуждением",
        "терпении в скорбях",
        "христианской семье",
        "венчании",
        "воспитании детей в вере",
        "Евхаристии и Причастии",
        "подготовке к исповеди",
        "духовной трезвенности",
        "благодарении Богу",
        "христианской надежде",
        "силе молитвы за усопших",
        "духовной жизни мирянина",
    ]

    templates = [
        "Подготовь проповедь о {topic}",
        "Сгенерируй православную проповедь о {topic}",
        "Напиши воскресную проповедь о {topic}",
    ]

    pool: List[str] = []
    for topic in feasts + saints + themes:
        for template in templates:
            pool.append(template.format(topic=topic))

    # Убираем дубликаты с сохранением порядка.
    seen = set()
    uniq: List[str] = []
    for p in pool:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def evaluate_one(service: OrthodoxAssistantService, prompt: str) -> Dict[str, object]:
    req = GenerateRequest(
        prompt=prompt,
        top_k_sources=6,
        max_new_tokens=260,
        temperature=0.78,
        top_p=0.92,
        repetition_penalty=1.12,
    )
    started = time.perf_counter()
    res = service.generate_sermon(req)
    elapsed = time.perf_counter() - started

    sermon = res.sermon
    low = sermon.lower()
    words = re.findall(r"[А-Яа-яA-Za-zЁё]+", sermon)
    has_structure = all(x in low for x in ["вступление.", "основная часть.", "заключение."])
    has_scripture_quote = "священное писание говорит:" in low
    has_father_quote = "наставляет: «" in low
    has_preacher_quote = "проповеднической традиции звучит слово" in low
    topic_covered = service._topic_is_covered(sermon, req)  # noqa: SLF001

    return {
        "prompt": prompt,
        "model_name": res.model_name,
        "seconds": round(elapsed, 3),
        "word_count": len(words),
        "has_structure": has_structure,
        "has_scripture_quote": has_scripture_quote,
        "has_father_quote": has_father_quote,
        "has_preacher_quote": has_preacher_quote,
        "topic_covered": topic_covered,
        "sermon_preview": sermon[:340],
        "sermon_norm": normalize_text(sermon),
    }


def run_eval(count: int, output: Path) -> Dict[str, object]:
    get_settings.cache_clear()
    settings = get_settings()
    service = OrthodoxAssistantService(settings)

    pool = build_prompt_pool()
    prompts = pool[:count]
    results = [evaluate_one(service, prompt) for prompt in prompts]

    duplicates = 0
    seen = set()
    for item in results:
        norm = item["sermon_norm"]
        if norm in seen:
            duplicates += 1
        else:
            seen.add(norm)

    summary = {
        "count": count,
        "avg_seconds": round(mean(float(x["seconds"]) for x in results), 3) if results else 0.0,
        "avg_word_count": round(mean(int(x["word_count"]) for x in results), 1) if results else 0.0,
        "structure_rate": round(sum(bool(x["has_structure"]) for x in results) / max(1, count), 4),
        "scripture_quote_rate": round(sum(bool(x["has_scripture_quote"]) for x in results) / max(1, count), 4),
        "father_quote_rate": round(sum(bool(x["has_father_quote"]) for x in results) / max(1, count), 4),
        "preacher_quote_rate": round(sum(bool(x["has_preacher_quote"]) for x in results) / max(1, count), 4),
        "topic_coverage_rate": round(sum(bool(x["topic_covered"]) for x in results) / max(1, count), 4),
        "duplicate_count": duplicates,
        "distinct_count": len(seen),
        "model_names": sorted({str(x["model_name"]) for x in results}),
    }

    bad_cases = [
        {
            "prompt": str(x["prompt"]),
            "has_structure": bool(x["has_structure"]),
            "topic_covered": bool(x["topic_covered"]),
            "has_scripture_quote": bool(x["has_scripture_quote"]),
            "has_father_quote": bool(x["has_father_quote"]),
            "has_preacher_quote": bool(x["has_preacher_quote"]),
            "preview": str(x["sermon_preview"]),
        }
        for x in results
        if not (
            bool(x["has_structure"])
            and bool(x["topic_covered"])
            and bool(x["has_scripture_quote"])
            and bool(x["has_father_quote"])
            and bool(x["has_preacher_quote"])
        )
    ]

    report = {
        "summary": summary,
        "bad_cases": bad_cases,
        "results": [{k: v for k, v in item.items() if k != "sermon_norm"} for item in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Пакетная проверка генерации православных проповедей.")
    parser.add_argument("--count", type=int, default=10, help="Количество промтов для проверки.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval/sermon_eval.json",
        help="Путь для json-отчета.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    report = run_eval(count=args.count, output=output)
    print("Evaluation complete")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report saved to: {output}")


if __name__ == "__main__":
    main()
