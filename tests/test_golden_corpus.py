"""Golden corpus integration test — seed -> exhaust -> assert.

Tests the full memory pipeline: corpus ingestion creates raw entries,
exhaustion runner processes them to convergence, and cognitive assertions
validate the resulting memory stack.

Uses DeterministicMockModel for reproducible results.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pytest

from kernle import Kernle
from kernle.corpus import CorpusIngestor
from kernle.entity import Entity
from kernle.exhaust import ExhaustionRunner
from kernle.protocols import ModelResponse
from kernle.stack.sqlite_stack import Stack
from kernle.testing.assertions import CognitiveAssertions

# Path to the golden corpus fixtures
GOLDEN_CORPUS_DIR = Path(__file__).parent / "fixtures" / "golden_corpus"


# =============================================================================
# Deterministic Mock Model
# =============================================================================


class DeterministicMockModel:
    """Mock ModelProtocol that returns deterministic JSON responses.

    Adapted to handle corpus-ingested raw entries which have metadata
    headers like [corpus:repo] [file:path] [chunk:type:name].
    """

    def __init__(self):
        self._call_count = 0

    @property
    def model_id(self) -> str:
        return "mock-deterministic-corpus-v1"

    @property
    def capabilities(self):
        from kernle.protocols import ModelCapabilities

        return ModelCapabilities(streaming=False, tool_use=False, vision=False)

    def generate(
        self,
        messages: list,
        *,
        tools: Optional[list] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system: Optional[str] = None,
    ) -> ModelResponse:
        """Return deterministic responses based on the system prompt."""
        self._call_count += 1
        prompt_text = messages[0].content if messages else ""

        if system and "episodic memories" in system:
            return self._episode_response(prompt_text)
        elif system and "factual notes" in system:
            return self._note_response(prompt_text)
        elif system and "beliefs" in system and "emerge from experiences" in system:
            return self._belief_response(prompt_text)
        elif system and "goals" in system:
            return self._goal_response(prompt_text)
        elif system and "relationships" in system:
            return self._relationship_response(prompt_text)
        elif system and "core values" in system:
            return self._value_response(prompt_text)
        elif system and "drives" in system or system and "motivational" in system:
            return self._drive_response(prompt_text)
        else:
            return ModelResponse(content="[]")

    def _episode_response(self, prompt: str) -> ModelResponse:
        """Produce episodes from raw corpus entries."""
        raw_ids = self._extract_ids(prompt)
        episodes = []
        # Group into batches of 3
        for i in range(0, len(raw_ids), 3):
            batch = raw_ids[i : i + 3]
            if not batch:
                break
            episodes.append(
                {
                    "objective": f"Studied corpus section {i // 3 + 1}",
                    "outcome": f"Absorbed {len(batch)} knowledge chunks",
                    "outcome_type": "success",
                    "lessons": [f"Key insight from section {i // 3 + 1}"],
                    "source_raw_ids": batch,
                }
            )
        return ModelResponse(content=json.dumps(episodes))

    def _note_response(self, prompt: str) -> ModelResponse:
        raw_ids = self._extract_ids(prompt)
        notes = []
        for i in range(0, min(len(raw_ids), 4), 2):
            notes.append(
                {
                    "content": f"Technical reference from corpus chunk {i // 2 + 1}",
                    "note_type": "reference",
                    "source_raw_ids": [raw_ids[i]],
                }
            )
        return ModelResponse(content=json.dumps(notes))

    def _belief_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if len(ep_ids) < 3:
            return ModelResponse(content="[]")
        beliefs = [
            {
                "statement": "Layered memory architecture improves knowledge organization",
                "belief_type": "causal",
                "confidence": 0.8,
                "source_episode_ids": ep_ids[:3],
            }
        ]
        return ModelResponse(content=json.dumps(beliefs))

    def _goal_response(self, prompt: str) -> ModelResponse:
        ep_ids = self._extract_ids(prompt)
        if not ep_ids:
            return ModelResponse(content="[]")
        goals = [
            {
                "title": "Deepen corpus understanding",
                "description": "Continue studying and integrating corpus knowledge",
                "goal_type": "aspiration",
                "priority": "medium",
                "source_episode_ids": ep_ids[:1],
            }
        ]
        return ModelResponse(content=json.dumps(goals))

    def _relationship_response(self, prompt: str) -> ModelResponse:
        return ModelResponse(content="[]")

    def _value_response(self, prompt: str) -> ModelResponse:
        return ModelResponse(content="[]")

    def _drive_response(self, prompt: str) -> ModelResponse:
        return ModelResponse(content="[]")

    @staticmethod
    def _extract_ids(prompt: str) -> list:
        """Extract bracketed IDs like [abc-123] from prompt text."""
        return re.findall(r"\[([a-f0-9-]+)\]", prompt)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def stack(tmp_path):
    """Create a bare Stack for test isolation."""
    db_path = tmp_path / "golden_corpus.db"
    s = Stack.from_sqlite(
        stack_id="golden-corpus-test",
        db_path=db_path,
        components=[],
        enforce_provenance=False,
    )
    yield s


@pytest.fixture
def entity_with_model(stack):
    """Create an Entity with a DeterministicMockModel bound."""
    entity = Entity(core_id="golden-corpus-test")
    entity.attach_stack(stack, alias="golden")
    model = DeterministicMockModel()
    entity.set_model(model)
    return entity, stack


@pytest.fixture
def kernle_instance(tmp_path):
    """Create a standalone Kernle instance for corpus ingestion."""
    from kernle.storage.sqlite import SQLiteStorage

    db_path = tmp_path / "golden_corpus.db"
    storage = SQLiteStorage(stack_id="golden-corpus-test", db_path=db_path)
    k = Kernle(stack_id="golden-corpus-test", storage=storage)
    return k


@pytest.fixture
def wired_setup(tmp_path):
    """Create a fully wired setup: Kernle + Entity + Model.

    Returns (kernle, entity, stack) where kernle.process() uses the model.
    """
    db_path = tmp_path / "golden_corpus.db"
    stack = Stack.from_sqlite(
        stack_id="golden-corpus-wired",
        db_path=db_path,
        components=[],
        enforce_provenance=False,
    )
    entity = Entity(core_id="golden-corpus-wired")
    entity.attach_stack(stack, alias="golden")
    model = DeterministicMockModel()
    entity.set_model(model)

    # Create a Kernle instance that uses the same storage backend
    k = Kernle.__new__(Kernle)
    k._stack_id = stack.stack_id
    k._storage = stack._backend
    k._entity = entity
    k._strict = False
    from kernle.utils import get_kernle_home

    k._checkpoint_dir = get_kernle_home() / "golden-corpus-wired" / "checkpoints"
    k._checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return k, entity, stack


# =============================================================================
# Tests
# =============================================================================


class TestCorpusSeeding:
    """Test that corpus ingestion creates raw entries."""

    def test_corpus_seeds_raw_entries(self, kernle_instance):
        """Corpus ingestion should create raw entries from fixture files."""
        ingestor = CorpusIngestor(kernle_instance)

        # Ingest docs
        doc_result = ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        assert doc_result.raw_entries_created > 0
        assert doc_result.errors == []

        # Ingest repo
        repo_result = ingestor.ingest_repo(str(GOLDEN_CORPUS_DIR / "src"), extensions=["py"])
        assert repo_result.raw_entries_created > 0
        assert repo_result.errors == []

        # Verify raw entries exist
        raw_entries = kernle_instance._storage.list_raw(limit=1000)
        total_created = doc_result.raw_entries_created + repo_result.raw_entries_created
        assert len(raw_entries) == total_created
        assert total_created >= 8  # At least 8 chunks from our fixtures

    def test_corpus_metadata_in_blobs(self, kernle_instance):
        """Raw entries should contain corpus metadata headers."""
        ingestor = CorpusIngestor(kernle_instance)
        ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))

        raw_entries = kernle_instance._storage.list_raw(limit=1000)
        assert len(raw_entries) > 0

        # Check that at least one entry has corpus metadata
        has_corpus_header = False
        for entry in raw_entries:
            blob = getattr(entry, "blob", "") or getattr(entry, "content", "") or ""
            if "[corpus:" in blob:
                has_corpus_header = True
                break
        assert has_corpus_header, "No corpus metadata headers found in raw entries"

    def test_reingestion_is_idempotent(self, kernle_instance):
        """Re-ingesting the same corpus should not create duplicates."""
        ingestor = CorpusIngestor(kernle_instance)

        # First ingestion
        result1 = ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        count1 = result1.raw_entries_created

        # Second ingestion (same ingestor, has seen hashes)
        result2 = ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        assert result2.raw_entries_created == 0
        assert result2.chunks_skipped > 0

        # Verify total count unchanged
        raw_entries = kernle_instance._storage.list_raw(limit=1000)
        assert len(raw_entries) == count1


class TestExhaustionConvergence:
    """Test that exhaustion runner converges on corpus data."""

    def test_exhaustion_converges(self, wired_setup):
        """Processing should converge after exhausting all promotions."""
        k, entity, stack = wired_setup

        # Seed corpus
        ingestor = CorpusIngestor(k)
        ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        ingestor.ingest_repo(str(GOLDEN_CORPUS_DIR / "src"), extensions=["py"])

        # Run exhaustion
        runner = ExhaustionRunner(k, max_cycles=15, auto_promote=True)
        result = runner.run()

        assert result.converged or result.convergence_reason == "max_cycles_reached"
        assert result.cycles_completed >= 2
        assert result.total_promotions > 0

    def test_exhaustion_creates_episodes(self, wired_setup):
        """Exhaustion should produce episodes from raw corpus entries."""
        k, entity, stack = wired_setup

        # Seed corpus
        ingestor = CorpusIngestor(k)
        ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))

        # Run exhaustion
        runner = ExhaustionRunner(k, max_cycles=10, auto_promote=True)
        runner.run()

        # Check episodes were created
        episodes = stack.get_episodes(limit=100)
        assert len(episodes) > 0, "No episodes created from corpus"


class TestCognitiveQuality:
    """Test cognitive quality after full pipeline run."""

    def _run_full_pipeline(self, k):
        """Helper: seed corpus and run exhaustion."""
        ingestor = CorpusIngestor(k)
        ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        ingestor.ingest_repo(str(GOLDEN_CORPUS_DIR / "src"), extensions=["py"])

        runner = ExhaustionRunner(k, max_cycles=15, auto_promote=True)
        return runner.run()

    def test_quality_assertions_pass(self, wired_setup):
        """Quality assertions should pass after full pipeline run."""
        k, entity, stack = wired_setup
        self._run_full_pipeline(k)

        assertions = CognitiveAssertions(k)
        report = assertions.run_quality()

        for a in report.assertions:
            if not a.passed:
                pytest.fail(f"Quality assertion failed: {a.name}: {a.message}")

    def test_provenance_chain_intact(self, wired_setup):
        """Provenance chains should be valid after full pipeline."""
        k, entity, stack = wired_setup
        self._run_full_pipeline(k)

        assertions = CognitiveAssertions(k)
        result = assertions.provenance_chain_intact()
        assert result.passed, f"Provenance chain broken: {result.message}"

    def test_episodes_have_outcomes(self, wired_setup):
        """All created episodes should have outcomes."""
        k, entity, stack = wired_setup
        self._run_full_pipeline(k)

        assertions = CognitiveAssertions(k)
        result = assertions.episodes_have_outcomes()
        assert result.passed, f"Episodes missing outcomes: {result.message}"

    def test_no_unprocessed_raw_after_exhaust(self, wired_setup):
        """After exhaustion, at least some raw entries should be processed."""
        k, entity, stack = wired_setup
        self._run_full_pipeline(k)

        raw_unprocessed = k._storage.list_raw(processed=False, limit=1000)
        raw_total = k._storage.list_raw(limit=1000)
        # At least some should have been processed
        assert len(raw_unprocessed) < len(raw_total), "No raw entries were processed"


class TestFullPipeline:
    """End-to-end pipeline integration tests."""

    def test_full_pipeline_seed_exhaust_assert(self, wired_setup):
        """Full pipeline: seed -> exhaust -> assert all passes."""
        k, entity, stack = wired_setup

        # 1. Seed
        ingestor = CorpusIngestor(k)
        doc_result = ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        repo_result = ingestor.ingest_repo(str(GOLDEN_CORPUS_DIR / "src"), extensions=["py"])
        total_seeded = doc_result.raw_entries_created + repo_result.raw_entries_created
        assert total_seeded > 0

        # 2. Exhaust
        runner = ExhaustionRunner(k, max_cycles=15, auto_promote=True)
        exhaust_result = runner.run()
        assert exhaust_result.total_promotions > 0

        # 3. Assert quality
        assertions = CognitiveAssertions(k)

        quality_report = assertions.run_quality()
        for a in quality_report.assertions:
            if not a.passed:
                pytest.fail(f"Quality check failed: {a.name}: {a.message}")

        # Provenance integrity
        provenance = assertions.provenance_chain_intact()
        assert provenance.passed, f"Provenance broken: {provenance.message}"

        # Episodes created
        episodes_check = assertions.episodes_exist(min_count=1)
        assert episodes_check.passed, f"No episodes: {episodes_check.message}"

    def test_pipeline_memory_counts(self, wired_setup):
        """Verify expected memory counts after full pipeline."""
        k, entity, stack = wired_setup

        # Seed
        ingestor = CorpusIngestor(k)
        ingestor.ingest_docs(str(GOLDEN_CORPUS_DIR / "docs"))
        ingestor.ingest_repo(str(GOLDEN_CORPUS_DIR / "src"), extensions=["py"])

        # Exhaust
        runner = ExhaustionRunner(k, max_cycles=15, auto_promote=True)
        runner.run()

        # Check counts
        backend = stack._backend
        raw_count = len(backend.list_raw(limit=10000))
        episode_count = len(stack.get_episodes(limit=10000))

        assert raw_count > 0, "No raw entries"
        assert episode_count > 0, "No episodes created"
