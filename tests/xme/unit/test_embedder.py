"""Unit tests for LocalEmbedder (works even without sentence-transformers)."""
import pytest
from xme.extraction.embedder import LocalEmbedder

_DIMS = 384


class TestLocalEmbedder:
    def test_embed_returns_list_of_floats(self):
        embedder = LocalEmbedder()
        vec = embedder.embed("some text about authentication")
        assert isinstance(vec, list)
        assert len(vec) == _DIMS
        assert all(isinstance(v, float) for v in vec)

    def test_embed_empty_string(self):
        embedder = LocalEmbedder()
        vec = embedder.embed("")
        assert len(vec) == _DIMS

    def test_embed_batch(self):
        embedder = LocalEmbedder()
        vecs = embedder.embed_batch(["text a", "text b", "text c"])
        assert len(vecs) == 3
        assert all(len(v) == _DIMS for v in vecs)

    def test_embed_batch_empty(self):
        embedder = LocalEmbedder()
        assert embedder.embed_batch([]) == []

    def test_cosine_similarity_same_vector(self):
        embedder = LocalEmbedder()
        vec = embedder.embed("auth uses JWT tokens")
        sim = LocalEmbedder.cosine_similarity(vec, vec)
        # Same vector should be 1.0 (or very close if not zero-vector)
        assert sim >= 0.99 or sim == 0.0  # 0.0 if fallback zero-vector

    def test_cosine_similarity_different_vectors(self):
        embedder = LocalEmbedder()
        v1 = embedder.embed("authentication with JWT tokens")
        v2 = embedder.embed("database indexing with B-trees")
        sim = LocalEmbedder.cosine_similarity(v1, v2)
        # Different topics should have lower similarity than identical
        same_sim = LocalEmbedder.cosine_similarity(v1, v1)
        if same_sim > 0:  # not zero-vector fallback
            assert sim < same_sim

    def test_cosine_similarity_empty_vectors(self):
        sim = LocalEmbedder.cosine_similarity([], [])
        assert sim == 0.0

    def test_cosine_similarity_mismatched_dims(self):
        sim = LocalEmbedder.cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0])
        assert sim == 0.0

    def test_dims_property(self):
        embedder = LocalEmbedder()
        assert embedder.dims == _DIMS
