#!/usr/bin/env python3
"""Массовая генерация CSV manifest из RoyalLib для большого православного корпуса."""

from __future__ import annotations

import argparse
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class AuthorRef:
    slug: str
    name: str
    url: str


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, str]] = []
        self._href: Optional[str] = None
        self._text: List[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            txt = " ".join("".join(self._text).split())
            self.links.append((self._href, txt))
            self._href = None
            self._text = []


def fetch_html(url: str, timeout: int = 12) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", "ignore")


def parse_authors_page_links(root_html: str) -> List[str]:
    p = LinkParser()
    p.feed(root_html)
    out = set()
    for href, _ in p.links:
        if not href or "authors-" not in href or not href.endswith(".html"):
            continue
        full = "https:" + href if href.startswith("//") else href
        if full.startswith("/"):
            full = "https://royallib.com" + full
        out.add(full)
    return sorted(out)


def extract_author_refs(page_html: str) -> List[AuthorRef]:
    p = LinkParser()
    p.feed(page_html)
    refs: List[AuthorRef] = []
    for href, name in p.links:
        if not href or "/author/" not in href or not href.endswith(".html"):
            continue
        full = "https:" + href if href.startswith("//") else href
        if full.startswith("/"):
            full = "https://royallib.com" + full
        slug = full.rsplit("/", 1)[-1].replace(".html", "")
        refs.append(AuthorRef(slug=slug, name=name or slug, url=full))
    return refs


def is_orthodox_author(slug: str, name: str) -> bool:
    slug_low = slug.lower()
    name_low = (name or "").lower()

    slug_kw = [
        "zlatoust",
        "feofan_zatvornik",
        "zatvornik_feofan",
        "feofilakt",
        "bryanchaninov",
        "kronshtad",
        "zadonsk",
        "optin",
        "damaskin",
        "svyatogorets",
        "isaak_sirin",
        "nikolay_serb",
        "ioann_lestv",
        "siluan_afonskiy",
        "tihon_zadonskiy",
        "makariy_egipet",
        "amvrosiy_optinskiy",
        "paisiy_svyatogorets",
        "kirill_ierusal",
        "aleksandriyskiy_kirill",
        "grigoriy_bogoslov",
        "afanasiy_velik",
        "ignatiy_bryanchaninov",
        "vasiliy_velik",
        "maksim_ispoved",
    ]
    if any(k in slug_low for k in slug_kw):
        return True

    title_markers = [
        "свт.",
        "свт ",
        "блж.",
        "прп.",
        "священномуч",
        "митрополит",
        "патриарх",
        "протоиерей",
        "архимандрит",
        "игумен",
        "старец",
    ]
    strong_name_markers = [
        "златоуст",
        "феофан затворник",
        "феофилакт болгар",
        "брянчан",
        "кронштадт",
        "задонск",
        "оптин",
        "сирин",
        "святогор",
        "богослов",
        "лествич",
        "дамаскин",
        "сербск",
        "афанасий великий",
        "кирилл александрий",
        "григорий богослов",
        "максим исповедник",
        "василий великий",
    ]
    has_title = any(k in name_low for k in title_markers)
    has_strong_name = any(k in name_low for k in strong_name_markers)
    return has_title or has_strong_name


def extract_books_for_author(author: AuthorRef, html: str) -> List[Tuple[str, str]]:
    p = LinkParser()
    p.feed(html)
    out: List[Tuple[str, str]] = []
    for href, text in p.links:
        if not href:
            continue
        full = "https:" + href if href.startswith("//") else href
        if full.startswith("/"):
            full = "https://royallib.com" + full
        if "/book/" not in full or not full.endswith(".html"):
            continue
        if f"/book/{author.slug}/" not in full:
            continue
        book_slug = full.rsplit("/", 1)[-1].replace(".html", "")
        title = " ".join((text or book_slug).split()) or book_slug.replace("_", " ")
        out.append((book_slug, title))
    return out


def is_religious_title(book_slug: str, title: str) -> bool:
    low = f"{book_slug} {title}".lower()
    positive = [
        "проповед",
        "поучен",
        "слово",
        "духов",
        "молитв",
        "покаян",
        "пост",
        "литург",
        "церк",
        "православ",
        "свят",
        "евангел",
        "христ",
        "спас",
        "таинств",
        "добротолюб",
        "пастыр",
        "исповед",
        "аскет",
        "житие",
        "гомили",
    ]
    negative = [
        "роман",
        "повест",
        "фантаст",
        "детектив",
        "триллер",
        "разведк",
        "шпион",
        "национализм",
        "интеллектуальный инсульт",
        "петр велик",
        "сказк",
        "приключен",
    ]
    if any(k in low for k in negative):
        return False
    return any(k in low for k in positive)


def category_for(book_slug: str, title: str) -> Optional[str]:
    low = f"{book_slug} {title}".lower()
    if any(
        k in low
        for k in [
            "толкован",
            "изъяснен",
            "объяснен",
            "на евангел",
            "на_евангел",
            "на_матф",
            "на_марк",
            "на_лук",
            "на_иоанн",
        ]
    ):
        return "commentaries"
    if is_religious_title(book_slug, title):
        return "sermons"
    return None


def build_manifest(
    out_csv: Path,
    max_authors: int,
    workers: int,
    timeout: int,
) -> Dict[str, int]:
    root = fetch_html("https://royallib.com/authors/", timeout=timeout)
    author_pages = parse_authors_page_links(root)

    refs: List[AuthorRef] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(fetch_html, url, timeout) for url in author_pages]
        for fut in as_completed(futs):
            try:
                html = fut.result()
            except Exception:
                continue
            refs.extend(extract_author_refs(html))

    uniq: Dict[str, AuthorRef] = {}
    for r in refs:
        uniq.setdefault(r.slug, r)

    candidates = [r for r in uniq.values() if is_orthodox_author(r.slug, r.name)]
    candidates.sort(key=lambda x: x.name.lower())
    if max_authors > 0:
        candidates = candidates[:max_authors]

    author_html: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        fut_map = {ex.submit(fetch_html, a.url, timeout): a for a in candidates}
        for fut in as_completed(fut_map):
            a = fut_map[fut]
            try:
                author_html[a.slug] = fut.result()
            except Exception:
                continue

    rows: List[dict] = []
    seen_urls = set()
    for a in candidates:
        html = author_html.get(a.slug)
        if not html:
            continue
        books = extract_books_for_author(a, html)
        for bslug, title in books:
            url = f"https://royallib.com/get/txt/{a.slug}/{bslug}.zip"
            if url in seen_urls:
                continue
            seen_urls.add(url)
            category = category_for(bslug, title)
            if not category:
                continue
            rows.append(
                {
                    "enabled": "1",
                    "slug": f"{a.slug}_{bslug}",
                    "category": category,
                    "title": title,
                    "author": a.name,
                    "reference": "Православная литература (RoyalLib)",
                    "url": url,
                }
            )

    rows.sort(key=lambda r: (r["category"], r["author"], r["title"]))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["enabled", "slug", "category", "title", "author", "reference", "url"])
        w.writeheader()
        w.writerows(rows)

    stats = {
        "author_pages": len(author_pages),
        "all_authors": len(uniq),
        "orth_candidates": len(candidates),
        "manifest_rows": len(rows),
        "commentaries": sum(1 for r in rows if r["category"] == "commentaries"),
        "sermons": sum(1 for r in rows if r["category"] == "sermons"),
    }
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="data/source_import/remote_sources.csv")
    parser.add_argument("--max-authors", type=int, default=220)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    out_csv = (project_root / args.out_csv).resolve()
    stats = build_manifest(out_csv, max_authors=args.max_authors, workers=args.workers, timeout=args.timeout)
    print(f"Manifest written: {out_csv}")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
