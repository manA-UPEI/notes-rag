"""Bare-metal vector store: a numpy array plus a metadata list. No database.

Similarity search is a single matrix-vector dot product because embeddings
are normalized at insert time, so cosine similarity reduces to a dot product.
"""

import json
from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self):
        self.vectors: np.ndarray | None = None
        self.metadata: list[dict] = []

    def add(self, vectors: np.ndarray, metadata: list[dict]):
        self.vectors = vectors
        self.metadata = metadata

    def search(self, query_vector: np.ndarray, top_k: int = 4) -> list[dict]:
        if self.vectors is None or len(self.metadata) == 0:
            return []
        scores: np.ndarray = self.vectors @ query_vector
        top_indices = np.argsort(-scores)[:top_k]
        return [
            {**self.metadata[i], "score": float(scores[i])} for i in top_indices
        ]

    def save(self, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)
        np.save(index_dir / "vectors.npy", self.vectors)
        (index_dir / "metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        store = cls()
        store.vectors = np.load(index_dir / "vectors.npy")
        store.metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
        return store
