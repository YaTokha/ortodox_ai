"""Подготовка корпуса для дообучения GPT-2.

Ожидаемая структура:
- data/raw/bible/*.txt
- data/raw/commentaries/*.txt
- data/raw/sermons/*.txt

Каждый txt-файл можно начинать метаданными:
# title: ...
# author: ...
# reference: ...

После пустой строки идет основной текст.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass
class Doc:
    id: str
    source_type: str
    title: str
    author: str
    reference: str
    text: str


def normalize(text: str) -> str:
    return " ".join(text.replace("\u00a0", " ").split()).strip()


def parse_txt(path: Path, source_type: str, doc_id: str) -> Doc:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = raw.splitlines()

    meta: Dict[str, str] = {"title": path.stem, "author": "unknown", "reference": ""}
    body_start = 0

    for i, line in enumerate(lines):
        if not line.strip():
            body_start = i + 1
            break
        if line.startswith("#") and ":" in line:
            key, val = line[1:].split(":", 1)
            key = key.strip().lower()
            if key in meta:
                meta[key] = val.strip()

    body = "\n".join(lines[body_start:]) if body_start > 0 else raw
    return Doc(
        id=doc_id,
        source_type=source_type,
        title=meta["title"],
        author=meta["author"],
        reference=meta["reference"],
        text=normalize(body),
    )


def chunk_text(text: str, max_chars: int = 1200) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    def split_long_piece(piece: str) -> List[str]:
        if len(piece) <= max_chars:
            return [piece]
        sentences = re.split(r"(?<=[.!?])\s+", piece)
        if len(sentences) <= 1:
            # fallback: режем по окнам фиксированной длины
            return [piece[i : i + max_chars] for i in range(0, len(piece), max_chars)]

        out: List[str] = []
        cur = ""
        for sent in sentences:
            candidate = (cur + " " + sent).strip() if cur else sent
            if len(candidate) <= max_chars:
                cur = candidate
            else:
                if cur:
                    out.append(cur)
                if len(sent) > max_chars:
                    out.extend([sent[i : i + max_chars] for i in range(0, len(sent), max_chars)])
                    cur = ""
                else:
                    cur = sent
        if cur:
            out.append(cur)
        return out

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        paragraphs = [text]

    chunks: List[str] = []
    current = ""
    for p in paragraphs:
        if len(p) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(split_long_piece(p))
            continue
        candidate = (current + "\n" + p).strip() if current else p
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)

    return chunks


def iter_txt_files(raw_root: Path) -> Iterable[Tuple[Path, str]]:
    source_dirs = {
        "bible": "bible",
        "commentaries": "commentary",
        "sermons": "sermon",
    }
    for folder, source_type in source_dirs.items():
        p = raw_root / folder
        if not p.exists():
            continue
        for file in sorted(p.rglob("*.txt")):
            yield file, source_type


def split_dataset(rows: List[dict], seed: int = 42) -> Tuple[List[dict], List[dict], List[dict]]:
    random.Random(seed).shuffle(rows)
    n = len(rows)
    if n < 3:
        # Для минимального корпуса: оставляем всё в train, чтобы не падал пайплайн.
        return rows, [], []

    n_valid = max(1, int(n * 0.1))
    n_test = max(1, int(n * 0.1))
    n_train = n - n_valid - n_test
    if n_train < 1:
        n_train = 1
        n_valid = 1
        n_test = n - n_train - n_valid

    train = rows[:n_train]
    valid = rows[n_train : n_train + n_valid]
    test = rows[n_train + n_valid : n_train + n_valid + n_test]
    return train, valid, test


def deduplicate_rows(rows: List[dict]) -> List[dict]:
    seen = set()
    out: List[dict] = []
    for row in rows:
        key = normalize(row["text"]).lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def parse_ratio_spec(spec: str) -> Dict[str, float]:
    """
    Формат:
      "bible=0.2,commentary=0.35,sermon=0.45"
    """
    mapping: Dict[str, str] = {
        "bible": "bible",
        "commentary": "commentary",
        "commentaries": "commentary",
        "sermon": "sermon",
        "sermons": "sermon",
    }
    out: Dict[str, float] = {}
    for chunk in (spec or "").split(","):
        part = chunk.strip()
        if not part or "=" not in part:
            continue
        raw_k, raw_v = part.split("=", 1)
        k = mapping.get(raw_k.strip().lower())
        if not k:
            continue
        try:
            v = float(raw_v.strip())
        except ValueError:
            continue
        if v > 0:
            out[k] = v
    return out


def count_by_source(rows: List[dict]) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    for row in rows:
        src = row.get("source_type", "unknown")
        stats[src] = stats.get(src, 0) + 1
    return stats


def rebalance_rows(
    rows: List[dict],
    ratio_spec: str,
    seed: int = 42,
    max_total_rows: int = 0,
) -> List[dict]:
    if not rows:
        return rows

    groups: Dict[str, List[dict]] = {}
    for row in rows:
        groups.setdefault(row["source_type"], []).append(row)
    source_types = sorted(groups.keys())
    if len(source_types) <= 1:
        return rows

    desired = parse_ratio_spec(ratio_spec)
    if not desired:
        desired = {"bible": 0.25, "commentary": 0.35, "sermon": 0.40}

    # Нормализуем доли под реально присутствующие типы источников.
    weights: Dict[str, float] = {}
    known_sum = 0.0
    unknown_types = []
    for src in source_types:
        if src in desired:
            w = max(0.0, desired[src])
            weights[src] = w
            known_sum += w
        else:
            unknown_types.append(src)

    remaining = max(0.0, 1.0 - known_sum)
    if unknown_types:
        fill = remaining / len(unknown_types) if remaining > 0 else 1.0 / len(source_types)
        for src in unknown_types:
            weights[src] = fill

    total_w = sum(weights.values())
    if total_w <= 0:
        eq = 1.0 / len(source_types)
        for src in source_types:
            weights[src] = eq
    else:
        for src in source_types:
            weights[src] = weights[src] / total_w

    target_total = max_total_rows if max_total_rows > 0 else len(rows)
    float_targets = {src: target_total * weights[src] for src in source_types}
    target_counts = {src: int(math.floor(v)) for src, v in float_targets.items()}
    remainder = target_total - sum(target_counts.values())
    if remainder > 0:
        fractions = sorted(
            source_types,
            key=lambda src: float_targets[src] - target_counts[src],
            reverse=True,
        )
        for src in fractions[:remainder]:
            target_counts[src] += 1

    rnd = random.Random(seed)
    balanced: List[dict] = []
    for src in source_types:
        items = groups[src]
        target = target_counts[src]
        if target <= 0:
            continue
        if len(items) >= target:
            balanced.extend(rnd.sample(items, target))
        else:
            balanced.extend(items)
            balanced.extend(rnd.choice(items) for _ in range(target - len(items)))

    rnd.shuffle(balanced)
    return balanced


def write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="data/raw")
    parser.add_argument("--out-corpus", default="data/processed/corpus.jsonl")
    parser.add_argument("--out-train", default="data/processed/train.jsonl")
    parser.add_argument("--out-valid", default="data/processed/valid.jsonl")
    parser.add_argument("--out-test", default="data/processed/test.jsonl")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deduplicate", dest="deduplicate", action="store_true")
    parser.add_argument("--no-deduplicate", dest="deduplicate", action="store_false")
    parser.set_defaults(deduplicate=True)
    parser.add_argument("--balance-source-types", action="store_true")
    parser.add_argument("--target-ratios", default="bible=0.25,commentary=0.35,sermon=0.40")
    parser.add_argument("--max-total-rows", type=int, default=0)
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    rows: List[dict] = []
    idx = 1

    for file, source_type in iter_txt_files(raw_root):
        doc = parse_txt(file, source_type=source_type, doc_id=str(idx))
        idx += 1
        for chunk_i, chunk in enumerate(chunk_text(doc.text, args.max_chars), start=1):
            chunk = normalize(chunk)
            if len(chunk) < args.min_chars:
                continue
            rows.append(
                {
                    "id": f"{doc.id}_{chunk_i}",
                    "source_type": doc.source_type,
                    "title": doc.title,
                    "author": doc.author,
                    "reference": doc.reference,
                    "text": chunk,
                }
            )

    if not rows:
        raise SystemExit("Корпус пуст. Добавьте txt-файлы в data/raw/*")

    initial_count = len(rows)
    if args.deduplicate:
        rows = deduplicate_rows(rows)

    if args.balance_source_types:
        rows = rebalance_rows(
            rows,
            ratio_spec=args.target_ratios,
            seed=args.seed,
            max_total_rows=args.max_total_rows,
        )

    train_rows, valid_rows, test_rows = split_dataset(rows, seed=args.seed)

    write_jsonl(Path(args.out_corpus), rows)
    write_jsonl(Path(args.out_train), train_rows)
    write_jsonl(Path(args.out_valid), valid_rows)
    write_jsonl(Path(args.out_test), test_rows)

    print(f"Всего примеров (до фильтрации): {initial_count}")
    print(f"Всего примеров (финально): {len(rows)}")
    print(f"Распределение по типам: {count_by_source(rows)}")
    print(f"Train/Valid/Test: {len(train_rows)}/{len(valid_rows)}/{len(test_rows)}")
    print(f"Корпус сохранен: {args.out_corpus}")


if __name__ == "__main__":
    main()
