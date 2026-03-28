#!/usr/bin/env python3
"""Импорт расширенного корпуса для ВКР.

Что делает скрипт:
1) Импортирует Библию из локального архива (опционально).
2) Загружает удалённые источники (RoyalLib и др.) из встроенного списка + CSV manifest.
3) Импортирует локальные TXT/ZIP из `data/source_import/manual/**`.

Цель: быстро собрать большой корпус проповедей/толкований для дообучения.
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMPORT_DIR = PROJECT_ROOT / "data" / "source_import"

DEFAULT_BIBLE_ARCHIVE = Path("/Users/tuhtaevtahir/Downloads/8_Bible_txt.zip")
BIBLE_EXTRACTED_FILE = IMPORT_DIR / "Bible_txt.txt"
DEFAULT_MANIFEST = IMPORT_DIR / "remote_sources.csv"
MANUAL_IMPORT_DIR = IMPORT_DIR / "manual"


@dataclass
class RemoteText:
    slug: str
    category: str  # bible/commentaries/sermons
    title: str
    author: str
    reference: str
    url: str


BASE_REMOTE_TEXTS: List[RemoteText] = [
    RemoteText(
        slug="zlatoust_matthew",
        category="commentaries",
        title="Толкование на святого Матфея Евангелиста",
        author="Свт. Иоанн Златоуст",
        reference="Толкование Евангелия",
        url="https://royallib.com/get/txt/zlatoust_sv_ioann/tolkovanie_na_svyatogo_matfeya_evangelista.zip",
    ),
    RemoteText(
        slug="feofilakt_luke",
        category="commentaries",
        title="Толкование на Евангелие от Луки",
        author="Блж. Феофилакт Болгарский",
        reference="Толкование Евангелия",
        url="https://royallib.com/get/txt/feofilakt_blg/tolkovanie_na_evangelie_ot_luki.zip",
    ),
    RemoteText(
        slug="feofilakt_matthew",
        category="commentaries",
        title="Толкование на Евангелие от Матфея",
        author="Блж. Феофилакт Болгарский",
        reference="Толкование Евангелия",
        url="https://royallib.com/get/txt/feofilakt_blg/tolkovanie_na_evangelie_ot_matfeya.zip",
    ),
    RemoteText(
        slug="feofilakt_mark",
        category="commentaries",
        title="Толкование на Евангелие от Марка",
        author="Блж. Феофилакт Болгарский",
        reference="Толкование Евангелия",
        url="https://royallib.com/get/txt/feofilakt_blg/tolkovanie_na_evangelie_ot_marka.zip",
    ),
    RemoteText(
        slug="surozh_sunday_sermons",
        category="sermons",
        title="Воскресные проповеди",
        author="Митрополит Антоний Сурожский",
        reference="Проповеди",
        url="https://royallib.com/get/txt/surogskiy__mitropolit/voskresnie_propovedi.zip",
    ),
    RemoteText(
        slug="dimitriy_sermons_1",
        category="sermons",
        title="Проповеди 1",
        author="Протоиерей Димитрий",
        reference="Проповеди",
        url="https://royallib.com/get/txt/dimitriy_protoierey/propovedi_1.zip",
    ),
    RemoteText(
        slug="ioann_kronshtadtsky_prayer",
        category="sermons",
        title="О молитве (выборки из писаний)",
        author="Св. Иоанн Кронштадтский",
        reference="Духовные наставления",
        url="https://royallib.com/get/txt/ioann_kronshtadtskiy/o_molitve_viborki_iz_ego_pisaniy.zip",
    ),
]


def ensure_dirs() -> None:
    (RAW_DIR / "bible").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "commentaries").mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "sermons").mkdir(parents=True, exist_ok=True)
    IMPORT_DIR.mkdir(parents=True, exist_ok=True)
    MANUAL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_royallib_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    marker = "Приятного чтения!"
    if marker in text:
        text = text.split(marker, 1)[1]

    lines = text.split("\n")
    cleaned = []
    for line in lines:
        l = line.strip()
        if not l:
            cleaned.append("")
            continue
        l_low = l.lower()
        if "royallib.ru" in l_low or "royallib.com" in l_low:
            continue
        if l.startswith("Все книги автора:") or l.startswith("Эта же книга"):
            continue
        if l.startswith("Спасибо, что скачали"):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    return normalize_spaces(text)


def try_decode(data: bytes) -> str:
    for enc in ("utf-8", "cp1251", "windows-1251", "koi8-r", "cp866"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("cp1251", errors="ignore")


def sanitize_filename(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-zа-я0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")[:120]


def write_raw_file(target: Path, title: str, author: str, reference: str, body: str) -> None:
    content = (
        f"# title: {title}\n"
        f"# author: {author}\n"
        f"# reference: {reference}\n\n"
        f"{body.strip()}\n"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def extract_bible_archive(bible_archive: Path) -> None:
    if BIBLE_EXTRACTED_FILE.exists() and BIBLE_EXTRACTED_FILE.stat().st_size > 1000:
        return
    if not bible_archive.exists():
        raise FileNotFoundError(f"Не найден файл: {bible_archive}")

    # Архив под именем zip может быть RAR, bsdtar справляется.
    import subprocess

    subprocess.run(["bsdtar", "-xf", str(bible_archive), "-C", str(IMPORT_DIR)], check=True)


def split_bible_by_books(full_text: str) -> Iterable[Tuple[str, str]]:
    pattern = re.compile(r"^==\s*([^=].*?)\s*==\s*$", flags=re.M)
    matches = list(pattern.finditer(full_text))

    for idx, m in enumerate(matches):
        title = m.group(1).strip()
        if not title or len(title) < 2:
            continue

        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if len(body) < 150:
            continue

        body = re.sub(r"^===\s*\d+\s*===\s*$", "", body, flags=re.M)
        body = normalize_spaces(body)
        if len(body) < 150:
            continue

        yield title, body


def import_bible(bible_archive: Path) -> int:
    extract_bible_archive(bible_archive)
    raw_bytes = BIBLE_EXTRACTED_FILE.read_bytes()
    text = try_decode(raw_bytes).replace("\r\n", "\n")

    out_dir = RAW_DIR / "bible"
    count = 0
    for i, (title, body) in enumerate(split_bible_by_books(text), start=1):
        fname = f"bible_{i:03d}_{sanitize_filename(title)}.txt"
        write_raw_file(
            out_dir / fname,
            title=title,
            author="Священное Писание",
            reference="Синодальный текст, источник: локальный архив Bible_txt.txt",
            body=body,
        )
        count += 1
    return count


def read_main_text_from_zip(data: bytes) -> str:
    zf = zipfile.ZipFile(io.BytesIO(data))
    txt_infos = [i for i in zf.infolist() if i.filename.lower().endswith(".txt")]
    if not txt_infos:
        raise RuntimeError("В архиве нет TXT файла")
    main = max(txt_infos, key=lambda x: x.file_size)
    return try_decode(zf.read(main))


def download(url: str, timeout: int) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ortodox-ai-corpus-ingest/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def normalize_category(category: str) -> Optional[str]:
    c = (category or "").strip().lower()
    mapping = {
        "bible": "bible",
        "commentary": "commentaries",
        "commentaries": "commentaries",
        "sermon": "sermons",
        "sermons": "sermons",
    }
    return mapping.get(c)


def load_manifest_csv(path: Path) -> List[RemoteText]:
    if not path.exists():
        return []
    rows: List[RemoteText] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            enabled = (raw.get("enabled", "1") or "1").strip().lower()
            if enabled in {"0", "false", "no"}:
                continue
            category = normalize_category(raw.get("category", ""))
            if not category:
                continue
            url = (raw.get("url") or "").strip()
            slug = (raw.get("slug") or "").strip()
            if not url or not slug:
                continue
            rows.append(
                RemoteText(
                    slug=slug,
                    category=category,
                    title=(raw.get("title") or slug).strip(),
                    author=(raw.get("author") or "не указан").strip(),
                    reference=(raw.get("reference") or "православный источник").strip(),
                    url=url,
                )
            )
    return rows


def unique_remote_sources(items: List[RemoteText]) -> List[RemoteText]:
    seen = set()
    out: List[RemoteText] = []
    for item in items:
        key = (item.slug, item.url)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def read_local_text_file(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix.lower() == ".zip" or zipfile.is_zipfile(path):
        return read_main_text_from_zip(data)
    return try_decode(data)


def parse_optional_meta(text: str, fallback_title: str) -> Tuple[str, str, str, str]:
    lines = text.replace("\r\n", "\n").splitlines()
    meta: Dict[str, str] = {"title": fallback_title, "author": "не указан", "reference": "локальный импорт"}
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
    body = "\n".join(lines[body_start:]) if body_start > 0 else text
    return meta["title"], meta["author"], meta["reference"], normalize_spaces(body)


def import_local_manual(min_chars: int = 1000) -> Dict[str, int]:
    stats = {"bible": 0, "commentaries": 0, "sermons": 0}
    if not MANUAL_IMPORT_DIR.exists():
        return stats

    for path in sorted(MANUAL_IMPORT_DIR.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".txt", ".zip"}:
            continue

        parent_names = {p.name.lower() for p in path.parents}
        if "bible" in parent_names:
            category = "bible"
        elif "commentaries" in parent_names or "commentary" in parent_names:
            category = "commentaries"
        else:
            category = "sermons"

        raw_text = read_local_text_file(path)
        title, author, reference, body = parse_optional_meta(raw_text, fallback_title=path.stem)
        body = clean_royallib_text(body)
        if len(body) < min_chars:
            continue

        target = RAW_DIR / category / f"{category[:-1] if category.endswith('s') else category}_{sanitize_filename(path.stem)}.txt"
        write_raw_file(target, title=title, author=author, reference=reference, body=body)
        stats[category] += 1
    return stats


def import_remote_texts(
    items: List[RemoteText],
    timeout: int,
    min_chars: int,
    continue_on_error: bool,
    workers: int,
) -> Dict[str, int]:
    stats = {"commentaries": 0, "sermons": 0, "failed": 0}

    def handle_item(item: RemoteText) -> str:
        try:
            data = download(item.url, timeout=timeout)
            text = read_main_text_from_zip(data) if zipfile.is_zipfile(io.BytesIO(data)) else try_decode(data)
            clean_text = clean_royallib_text(text)
            if len(clean_text) < min_chars:
                return "skipped"

            target = RAW_DIR / item.category / f"{item.category[:-1]}_{sanitize_filename(item.slug)}.txt"
            reference = f"{item.reference}; источник: {item.url}"
            write_raw_file(target, item.title, item.author, reference, clean_text)
            return item.category
        except Exception as exc:
            if not continue_on_error:
                raise
            print(f"[WARN] Не удалось импортировать {item.url}: {exc}")
            return "failed"

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        fut_map = {ex.submit(handle_item, item): item for item in items}
        for fut in as_completed(fut_map):
            status = fut.result()
            if status in ("commentaries", "sermons"):
                stats[status] += 1
            elif status == "failed":
                stats["failed"] += 1
    return stats


def cleanup_old_examples() -> None:
    for folder in [RAW_DIR / "bible", RAW_DIR / "commentaries", RAW_DIR / "sermons"]:
        for p in folder.glob("example_*.txt"):
            p.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bible-archive", default=str(DEFAULT_BIBLE_ARCHIVE))
    parser.add_argument("--skip-bible", action="store_true")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--only-manifest", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--min-remote-chars", type=int, default=1000)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    cleanup_old_examples()

    bible_count = 0
    if not args.skip_bible:
        bible_count = import_bible(Path(args.bible_archive))

    manifest_items = load_manifest_csv(Path(args.manifest))
    remote_items = manifest_items if args.only_manifest else unique_remote_sources(BASE_REMOTE_TEXTS + manifest_items)

    remote_stats = import_remote_texts(
        remote_items,
        timeout=args.timeout,
        min_chars=args.min_remote_chars,
        continue_on_error=args.continue_on_error,
        workers=args.workers,
    )
    local_stats = import_local_manual(min_chars=args.min_remote_chars)

    print(f"Bible files imported: {bible_count}")
    print(
        "Remote imported: "
        f"commentaries={remote_stats['commentaries']}, sermons={remote_stats['sermons']}, failed={remote_stats['failed']}"
    )
    print(
        "Local manual imported: "
        f"bible={local_stats['bible']}, commentaries={local_stats['commentaries']}, sermons={local_stats['sermons']}"
    )
    print(f"Raw corpus path: {RAW_DIR}")


if __name__ == "__main__":
    main()
