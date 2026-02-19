"""Tests for EmbeddingComponent (StackComponentProtocol implementation)."""

from unittest.mock import MagicMock

import pytest

from kernle.protocols import InferenceService, SearchResult, StackComponentProtocol
from kernle.stack.components.embedding import EmbeddingComponent
from kernle.storage.embeddings import HASH_EMBEDDING_DIM

# ---- Fixtures ----


def _make_mock_inference(embed_result=None, dimension=384, provider_id="test-embed"):
    """Create a mock InferenceService."""
    inference = MagicMock(spec=InferenceService)
    inference.embedding_dimension = dimension
    inference.embedding_provider_id = provider_id
    if embed_result is not None:
        inference.embed.return_value = embed_result
        inference.embed_batch.return_value = [embed_result]
    else:
        inference.embed.return_value = [0.1] * dimension
        inference.embed_batch.return_value = [[0.1] * dimension]
    return inference


# ---- Protocol Conformance ----


class TestProtocolConformance:
    """Test that EmbeddingComponent conforms to StackComponentProtocol."""

    def test_is_stack_component(self):
        comp = EmbeddingComponent()
        assert isinstance(comp, StackComponentProtocol)

    def test_name_property(self):
        comp = EmbeddingComponent()
        assert comp.name == "embedding-ngram"

    def test_version_property(self):
        comp = EmbeddingComponent()
        assert comp.version == "1.0.0"

    def test_required_is_true(self):
        comp = EmbeddingComponent()
        assert comp.required is True

    def test_needs_inference_is_false(self):
        comp = EmbeddingComponent()
        assert comp.needs_inference is False


# ---- Lifecycle ----


class TestLifecycle:
    """Test attach/detach/set_inference lifecycle."""

    def test_attach_sets_stack_id(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        assert comp._stack_id == "stack-1"

    def test_attach_sets_inference(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference()
        comp.attach("stack-1", inference=inference)
        assert comp._inference is inference

    def test_attach_without_inference(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        assert comp._inference is None

    def test_detach_clears_state(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1", inference=_make_mock_inference())
        comp.detach()
        assert comp._stack_id is None
        assert comp._inference is None

    def test_set_inference_updates(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        inference = _make_mock_inference()
        comp.set_inference(inference)
        assert comp._inference is inference

    def test_set_inference_to_none(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1", inference=_make_mock_inference())
        comp.set_inference(None)
        assert comp._inference is None


# ---- Hooks ----


class TestOnSave:
    """Test on_save hook."""

    def test_on_save_returns_none(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        result = comp.on_save("episode", "ep-1", {"objective": "test"})
        assert result is None


class TestOnSearch:
    """Test on_search hook."""

    def test_on_search_returns_results_unchanged(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        results = [
            SearchResult(
                memory_type="episode",
                memory_id="ep-1",
                content="test",
                score=0.9,
            ),
            SearchResult(
                memory_type="belief",
                memory_id="b-1",
                content="belief",
                score=0.7,
            ),
        ]
        out = comp.on_search("query", results)
        assert out is results
        assert len(out) == 2

    def test_on_search_empty_results(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        out = comp.on_search("query", [])
        assert out == []


class TestOnLoad:
    """Test on_load hook."""

    def test_on_load_returns_none(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        result = comp.on_load({"values": []})
        assert result is None


class TestOnMaintenance:
    """Test on_maintenance hook."""

    def test_on_maintenance_returns_stats(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        stats = comp.on_maintenance()
        assert isinstance(stats, dict)
        assert stats["provider"] == "ngram-v1"
        assert stats["dimension"] == HASH_EMBEDDING_DIM
        assert stats["has_inference"] is False

    def test_on_maintenance_with_inference(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference(provider_id="model-embed-v1")
        comp.attach("stack-1", inference=inference)
        stats = comp.on_maintenance()
        assert stats["has_inference"] is True
        assert stats["provider"] == "model-embed-v1"


# ---- Embedding Interface ----


class TestEmbed:
    """Test embed() method."""

    def test_embed_without_inference_uses_local(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        result = comp.embed("hello world")
        assert isinstance(result, list)
        assert len(result) == HASH_EMBEDDING_DIM

    def test_embed_is_deterministic(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        r1 = comp.embed("hello world")
        r2 = comp.embed("hello world")
        assert r1 == r2

    def test_embed_with_inference_delegates(self):
        comp = EmbeddingComponent()
        expected = [0.5] * 384
        inference = _make_mock_inference(embed_result=expected)
        comp.attach("stack-1", inference=inference)
        result = comp.embed("hello world")
        assert result == expected
        inference.embed.assert_called_once_with("hello world")

    def test_embed_falls_back_on_inference_failure(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference()
        inference.embed.side_effect = RuntimeError("model down")
        comp.attach("stack-1", inference=inference)
        result = comp.embed("hello world")
        # Should still get a valid embedding from local fallback
        assert isinstance(result, list)
        assert len(result) == HASH_EMBEDDING_DIM


class TestEmbedBatch:
    """Test embed_batch() method."""

    def test_embed_batch_without_inference(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        results = comp.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(len(r) == HASH_EMBEDDING_DIM for r in results)

    def test_embed_batch_with_inference_delegates(self):
        comp = EmbeddingComponent()
        expected = [[0.5] * 384, [0.6] * 384]
        inference = _make_mock_inference()
        inference.embed_batch.return_value = expected
        comp.attach("stack-1", inference=inference)
        result = comp.embed_batch(["hello", "world"])
        assert result == expected

    def test_embed_batch_falls_back_on_inference_failure(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference()
        inference.embed_batch.side_effect = RuntimeError("model down")
        comp.attach("stack-1", inference=inference)
        results = comp.embed_batch(["hello", "world"])
        assert len(results) == 2
        assert all(len(r) == HASH_EMBEDDING_DIM for r in results)

    def test_embed_batch_empty_list(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1")
        results = comp.embed_batch([])
        assert results == []


# ---- Properties ----


class TestProperties:
    """Test embedding_dimension and embedding_provider_id."""

    def test_dimension_without_inference(self):
        comp = EmbeddingComponent()
        assert comp.embedding_dimension == HASH_EMBEDDING_DIM

    def test_dimension_with_inference(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference(dimension=768)
        comp.attach("stack-1", inference=inference)
        assert comp.embedding_dimension == 768

    def test_provider_id_without_inference(self):
        comp = EmbeddingComponent()
        assert comp.embedding_provider_id == "ngram-v1"

    def test_provider_id_with_inference(self):
        comp = EmbeddingComponent()
        inference = _make_mock_inference(provider_id="sentence-transformers/all-MiniLM-L6-v2")
        comp.attach("stack-1", inference=inference)
        assert comp.embedding_provider_id == "sentence-transformers/all-MiniLM-L6-v2"


# ---- Graceful Degradation ----


class TestGracefulDegradation:
    """Test that the component works correctly without inference."""

    def test_works_without_attach(self):
        """Embedding works even before attach (standalone use)."""
        comp = EmbeddingComponent()
        result = comp.embed("hello")
        assert isinstance(result, list)
        assert len(result) == HASH_EMBEDDING_DIM

    def test_works_after_detach(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1", inference=_make_mock_inference())
        comp.detach()
        result = comp.embed("hello")
        assert isinstance(result, list)
        assert len(result) == HASH_EMBEDDING_DIM

    def test_works_after_inference_removed(self):
        comp = EmbeddingComponent()
        comp.attach("stack-1", inference=_make_mock_inference())
        comp.set_inference(None)
        result = comp.embed("hello")
        assert isinstance(result, list)
        assert len(result) == HASH_EMBEDDING_DIM


# ---- Stack Integration ----


class TestStackIntegration:
    """Test EmbeddingComponent with Stack's component system."""

    def test_add_component_to_stack(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)
        assert "embedding-ngram" in stack.components
        assert stack.get_component("embedding-ngram") is comp

    def test_cannot_remove_required_component(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)
        with pytest.raises(ValueError, match="Cannot remove required"):
            stack.remove_component("embedding-ngram")

    def test_maintenance_includes_component(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)
        results = stack.maintenance()
        assert "embedding-ngram" in results
        assert results["embedding-ngram"]["provider"] == "ngram-v1"

    def test_on_attach_propagates_inference(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)

        inference = _make_mock_inference()
        stack.on_attach("core-1", inference=inference)
        assert comp._inference is inference

    def test_on_model_changed_propagates(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)

        inference1 = _make_mock_inference(provider_id="v1")
        stack.on_attach("core-1", inference=inference1)
        assert comp._inference is inference1

        inference2 = _make_mock_inference(provider_id="v2")
        stack.on_model_changed(inference2)
        assert comp._inference is inference2

    def test_on_detach_clears_inference(self, tmp_path):
        from kernle.stack.sqlite_stack import Stack

        stack = Stack.from_sqlite(
            stack_id="test-stack",
            db_path=tmp_path / "test.db",
            components=[],
            enforce_provenance=False,
        )
        comp = EmbeddingComponent()
        stack.add_component(comp)

        inference = _make_mock_inference()
        stack.on_attach("core-1", inference=inference)
        stack.on_detach("core-1")
        assert comp._inference is None
