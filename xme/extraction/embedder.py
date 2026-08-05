"""Local embedding using sentence-transformers all-MiniLM-L6-v2.

384-dimensional embeddings, ~80MB download once, no API key needed.
Falls back to a zero-vector if sentence-transformers is not installed,
so tests and fallback mode don't require the dependency.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_DIMS = 384


class LocalEmbedder:
    """Lazy-loaded sentence-transformers embedder.

    Usage::

        embedder = LocalEmbedder()
        vec = embedder.embed("some text")          # list[float] len=384
        vecs = embedder.embed_batch(["a", "b"])    # list[list[float]]
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = model_name
        self._model: Optional[object] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            logger.debug("LocalEmbedder loaded model %s", self._model_name)
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers. "
                "Using zero-vector fallback."
            )
            self._model = None

    def embed(self, text: str) -> list[float]:
        self._load()
        if self._model is None:
            return [0.0] * _DIMS
        try:
            vec = self._model.encode(text, normalize_embeddings=True)  # type: ignore[union-attr]
            return vec.tolist()
        except Exception as e:
            logger.warning("Embedding failed: %s", e)
            return [0.0] * _DIMS

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        if self._model is None:
            return [[0.0] * _DIMS for _ in texts]
        try:
            vecs = self._model.encode(texts, normalize_embeddings=True, batch_size=32)  # type: ignore[union-attr]
            return [v.tolist() for v in vecs]
        except Exception as e:
            logger.warning("Batch embedding failed: %s", e)
            return [[0.0] * _DIMS for _ in texts]

    @property
    def dims(self) -> int:
        return _DIMS

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Fast cosine similarity for two unit-norm vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        # vectors are already normalized if from embed()
        return max(-1.0, min(1.0, dot))
