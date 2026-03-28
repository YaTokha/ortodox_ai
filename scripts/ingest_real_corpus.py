#!/usr/bin/env python3
"""Импорт реального корпуса для ВКР.

1) Берет Библию из локального архива (RAR под именем zip), декодирует cp1251,
   режет по книгам и сохраняет в data/raw/bible.
2) Скачивает православные толкования и проповеди с RoyalLib,
   очищает служебные блоки и сохраняет в data/raw/commentaries и data/raw/sermons.
"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMPORT_DIR = PROJECT_ROOT / "data" / "source_import"

BIBLE_ARCHIVE = Path("/Users/tuhtaevtahir/Downloads/8_Bible_txt.zip")
BIBLE_EXTRACTED_FILE = IMPORT_DIR / "Bible_txt.txt"


@dataclass
class RemoteText:
    slug: str
    category: str
    title: str
    author: str
    reference: str
    url: str


REMOTE_TEXTS: List[RemoteText] = [
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
        if "royallib.ru" in l.lower() or "royallib.com" in l.lower():
            continue
        if l.startswith("Все книги автора:") or l.startswith("Эта же книга"):
            continue
        if l.startswith("Спасибо, что скачали"):
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = normalize_spaces(text)
    return text


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
    target.write_text(content, encoding="utf-8")


def extract_bible_archive() -> None:
    if BIBLE_EXTRACTED_FILE.exists() and BIBLE_EXTRACTED_FILE.stat().st_size > 1000:
        return
    if not BIBLE_ARCHIVE.exists():
        raise FileNotFoundError(f"Не найден файл: {BIBLE_ARCHIVE}")

    # Архив на самом деле RAR, но bsdtar умеет читать.
    import subprocess

    subprocess.run(
        [
            "bsdtar",
            "-xf",
            str(BIBLE_ARCHIVE),
            "-C",
            str(IMPORT_DIR),
        ],
        check=True,
    )


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

        # Убираем маркер разделов глав вида === 1 ===
        body = re.sub(r"^===\s*\d+\s*===\s*$", "", body, flags=re.M)
        body = normalize_spaces(body)
        if len(body) < 150:
            continue

        yield title, body


def import_bible() -> int:
    extract_bible_archive()
    raw_bytes = BIBLE_EXTRACTED_FILE.read_bytes()
    text = try_decode(raw_bytes)
    text = text.replace("\r\n", "\n")

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


def download(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def import_remote_texts() -> Tuple[int, int]:
    commentary_count = 0
    sermon_count = 0

    for item in REMOTE_TEXTS:
        data = download(item.url)
        text = read_main_text_from_zip(data)
        clean_text = clean_royallib_text(text)
        if len(clean_text) < 1000:
            continue

        target_dir = RAW_DIR / item.category
        fname = f"{item.category[:-1]}_{sanitize_filename(item.slug)}.txt"
        reference = f"{item.reference}; источник: {item.url}"

        write_raw_file(target_dir / fname, item.title, item.author, reference, clean_text)

        if item.category == "commentaries":
            commentary_count += 1
        elif item.category == "sermons":
            sermon_count += 1

    return commentary_count, sermon_count


def cleanup_old_examples() -> None:
    for folder in [RAW_DIR / "bible", RAW_DIR / "commentaries", RAW_DIR / "sermons"]:
        for p in folder.glob("example_*.txt"):
            p.unlink(missing_ok=True)


def main() -> None:
    ensure_dirs()
    cleanup_old_examples()

    bible_count = import_bible()
    comm_count, serm_count = import_remote_texts()

    print(f"Bible files imported: {bible_count}")
    print(f"Commentary files imported: {comm_count}")
    print(f"Sermon files imported: {serm_count}")


if __name__ == "__main__":
    main()
