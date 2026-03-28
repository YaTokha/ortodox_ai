import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from app.schemas import Citation
from app.services.text_preprocessor import TextPreprocessor


@dataclass
class CorpusChunk:
    id: str
    source_type: str
    author: Optional[str]
    title: Optional[str]
    reference: Optional[str]
    text: str


class CorpusRetrievalService:
    def __init__(self, corpus_path: Path):
        self.corpus_path = corpus_path
        self.preprocessor = TextPreprocessor()
        self.chunks: List[CorpusChunk] = []
        self._vectorizer: Any = None
        self._matrix: Any = None
        self._cosine_similarity = None
        self._load()

    def _load(self) -> None:
        if not self.corpus_path.exists():
            return

        with self.corpus_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = self.preprocessor.normalize(row.get("text", ""))
                if len(text) < 40:
                    continue

                self.chunks.append(
                    CorpusChunk(
                        id=str(row.get("id", len(self.chunks) + 1)),
                        source_type=row.get("source_type", "unknown"),
                        author=row.get("author"),
                        title=row.get("title"),
                        reference=row.get("reference"),
                        text=text,
                    )
                )

        if not self.chunks:
            return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            # Retrieval отключается, если sklearn не установлен.
            return

        corpus_texts = [c.text for c in self.chunks]
        self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=15000)
        self._matrix = self._vectorizer.fit_transform(corpus_texts)
        self._cosine_similarity = cosine_similarity

    def is_ready(self) -> bool:
        return self._vectorizer is not None and self._matrix is not None and len(self.chunks) > 0

    def search(self, query: str, top_k: int = 4) -> List[Citation]:
        if not self.is_ready():
            return []

        query = self.preprocessor.normalize(query)
        if not query:
            return []

        q_vec = self._vectorizer.transform([query])
        sim = self._cosine_similarity(q_vec, self._matrix).flatten()
        idx_sorted = sim.argsort()[::-1][:top_k]

        result = []
        for idx in idx_sorted:
            chunk = self.chunks[idx]
            excerpt = chunk.text[:420] + ("..." if len(chunk.text) > 420 else "")
            result.append(
                Citation(
                    id=chunk.id,
                    source_type=chunk.source_type,
                    author=chunk.author,
                    title=chunk.title,
                    reference=chunk.reference,
                    excerpt=excerpt,
                    score=float(sim[idx]),
                )
            )
        return result
