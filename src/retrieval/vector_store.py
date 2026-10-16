from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Any
import json
import re
import numpy as np
from rank_bm25 import BM25Okapi

from src.models import DocumentChunk
from src.retrieval.embeddings import EmbeddingGenerator


def cosine_similarity(a: list[float] | np.ndarray, b: list[float] | np.ndarray) -> float:
    """Compute Cosine Similarity between two vectors: dot(a, b) / (||a|| * ||b||)."""
    a_arr = np.array(a, dtype=np.float32)
    b_arr = np.array(b, dtype=np.float32)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def normalize_scores(scores: list[float]) -> list[float]:
    """Normalise raw scores to [0, 1] using min-max scaling."""
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 if max_s > 0 else 0.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def parse_version_key(v: Any) -> tuple:
    """Helper to convert a version string/number into a sortable tuple."""
    if v is None:
        return (0, 0)
    v_str = str(v).strip()
    if not v_str:
        return (0, 0)
    parts = re.split(r"[.\-_]", v_str)
    res = []
    for p in parts:
        if p.isdigit():
            res.append(int(p))
        else:
            res.append(p)
    return (1, tuple(res))


@dataclass
class VectorStore:
    """
    Unified Knowledge & Vector Store with built-in Hybrid Search (BM25 + Cosine).
    Combines storage, indexing, and multi-mode search in a single class.
    """
    chunks: dict[str, DocumentChunk] = field(default_factory=dict)
    file_meta: dict[str, float] = field(default_factory=dict)  # file_name -> last_modified_timestamp
    _embedding_gen: Optional[EmbeddingGenerator] = field(default=None, repr=False)
    _chunk_keys: list[str] = field(default_factory=list, repr=False)
    _embeddings: list[list[float]] = field(default_factory=list, repr=False)
    _bm25_index: Optional[BM25Okapi] = field(default=None, repr=False)

    @property
    def embedding_gen(self) -> EmbeddingGenerator:
        """Lazy load EmbeddingGenerator if not already provided."""
        if self._embedding_gen is None:
            self._embedding_gen = EmbeddingGenerator()
        return self._embedding_gen

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple clean tokenization for BM25."""
        return re.findall(r"\w+", text.lower())

    def _rebuild_indices(self) -> None:
        """Synchronize both Vector index and BM25 index."""
        self._chunk_keys = list(self.chunks.keys())
        self._embeddings = []
        tokenized_corpus = []

        for cid in self._chunk_keys:
            chunk = self.chunks[cid]
            if chunk.embedding is not None:
                self._embeddings.append(chunk.embedding)
            else:
                self._embeddings.append([])
            tokenized_corpus.append(self._tokenize(chunk.content))

        if tokenized_corpus:
            self._bm25_index = BM25Okapi(tokenized_corpus)
        else:
            self._bm25_index = None

    def add_chunk(self, chunk: DocumentChunk, embedding: Optional[list[float]] = None) -> None:
        """Add a single chunk to the store."""
        if embedding is not None:
            chunk.embedding = embedding
        self.chunks[chunk.chunk_id] = chunk
        self._rebuild_indices()

    def add_chunks(
        self,
        chunks: list[DocumentChunk],
        embeddings: Optional[list[list[float]]] = None,
    ) -> None:
        """Add multiple chunks to the store with optional pre-computed embeddings."""
        for i, chunk in enumerate(chunks):
            if embeddings and i < len(embeddings):
                chunk.embedding = embeddings[i]
            self.chunks[chunk.chunk_id] = chunk
        self._rebuild_indices()


    def search(
        self,
        query: str,
        top_k: int = 4,
        mode: Literal["hybrid", "vector", "bm25"] = "hybrid",
        w_bm25: float = 0.3,
        w_vector: float = 0.7,
        year: Optional[str | int] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Search for top_k most relevant chunks using Hybrid, Vector, or BM25 mode.
        Supports version filtering:
        - If year is specified: only searches chunks matching that exact version/year.
        - If year is not specified: keeps only chunks belonging to the highest version for each doc_id.
        Returns: list of (DocumentChunk, score) sorted descending by relevance.
        """
        if not self.chunks:
            return []

        # 0. Version / Year filtering
        if year is not None and str(year).strip() and str(year).strip().lower() not in ("none", "null"):
            target_year = str(year).strip()
            valid_idx = [
                i for i, cid in enumerate(self._chunk_keys)
                if str(self.chunks[cid].metadata.get("version", "")).strip() == target_year
            ]
        else:
            # Find maximum version per doc_id across all indexed chunks
            max_versions: dict[str, tuple] = {}
            for chunk in self.chunks.values():
                d_id = chunk.doc_id or (chunk.metadata or {}).get("doc_id", "")
                v_k = parse_version_key((chunk.metadata or {}).get("version"))
                if d_id not in max_versions or v_k > max_versions[d_id]:
                    max_versions[d_id] = v_k

            valid_idx = [
                i for i, cid in enumerate(self._chunk_keys)
                if parse_version_key((self.chunks[cid].metadata or {}).get("version")) == max_versions.get(
                    self.chunks[cid].doc_id or (self.chunks[cid].metadata or {}).get("doc_id", "")
                )
            ]

        if not valid_idx:
            return []

        top_k = min(top_k, len(valid_idx))

        # 1. Pure BM25 Search
        if mode == "bm25":
            if not self._bm25_index:
                return []
            tokens = self._tokenize(query)
            scores = list(self._bm25_index.get_scores(tokens))
            valid_scores = [scores[i] for i in valid_idx]
            ranked_local = np.argsort(valid_scores)[::-1][:top_k]
            return [
                (self.chunks[self._chunk_keys[valid_idx[idx]]], float(valid_scores[idx]))
                for idx in ranked_local
            ]

        # 2. Pure Vector Search (Cosine Similarity)
        query_vec = self.embedding_gen.embed_query(query)
        raw_vector_scores = [
            cosine_similarity(query_vec, self._embeddings[i]) if self._embeddings[i] else 0.0
            for i in valid_idx
        ]

        if mode == "vector":
            ranked_local = np.argsort(raw_vector_scores)[::-1][:top_k]
            return [
                (self.chunks[self._chunk_keys[valid_idx[idx]]], float(raw_vector_scores[idx]))
                for idx in ranked_local
            ]

        # 3. Hybrid Search (BM25 + Cosine Fusion)
        tokens = self._tokenize(query)
        all_bm25 = (
            list(self._bm25_index.get_scores(tokens))
            if self._bm25_index
            else [0.0] * len(self._chunk_keys)
        )
        valid_bm25_scores = [all_bm25[i] for i in valid_idx]

        norm_bm25 = normalize_scores(valid_bm25_scores)
        norm_vector = normalize_scores(raw_vector_scores)

        hybrid_scores = [
            w_bm25 * b + w_vector * v for b, v in zip(norm_bm25, norm_vector)
        ]

        ranked_local = np.argsort(hybrid_scores)[::-1][:top_k]
        return [
            (self.chunks[self._chunk_keys[valid_idx[idx]]], float(hybrid_scores[idx]))
            for idx in ranked_local
        ]

    def remove_chunks_by_file(self, file_name: str) -> int:
        """Remove all chunks belonging to a specific file name."""
        to_del = [
            cid for cid, c in self.chunks.items()
            if c.metadata.get("file_name") == file_name
        ]
        for cid in to_del:
            del self.chunks[cid]
        if file_name in self.file_meta:
            del self.file_meta[file_name]
        if to_del:
            self._rebuild_indices()
        return len(to_del)

    def save_to_json(self, file_path: str | Path) -> None:
        """Persist chunks and embeddings to a JSON file."""
        data = {
            "file_meta": self.file_meta,
            "chunks": [chunk.to_dict() for chunk in self.chunks.values()]
        }
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load_from_json(cls, file_path: str | Path) -> "VectorStore":
        """Load from JSON and immediately rebuild both Vector & BM25 indices."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        store = cls()
        store.file_meta = data.get("file_meta", {})
        for c_dict in data.get("chunks", []):
            chunk = DocumentChunk.from_dict(c_dict)
            store.chunks[chunk.chunk_id] = chunk

        store._rebuild_indices()
        return store
