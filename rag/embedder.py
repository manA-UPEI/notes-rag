"""Local embeddings via sentence-transformers. Free, offline, no API key."""

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)

    def embed_one(self, text: str) -> np.ndarray:
        return self.model.encode([text], normalize_embeddings=True)[0]
