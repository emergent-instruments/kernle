"""Contract tests for StackProtocol.

Verifies that Stack conforms to the StackProtocol contract.
These tests use real Stack instances (not mocks) and exercise
the full protocol surface: write/read roundtrips, search, load,
meta-memory, composition hooks, and component registry.

Designed to be reusable: future stack implementations can subclass
the fixtures to run the same contract suite.
"""

import json
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, PropertyMock

import pytest

from kernle.protocols import (
    InferenceService,
    StackComponentProtocol,
)
from kernle.protocols import (
    SearchResult as ProtocolSearchResult,
)
from kernle.types import (
    Belief,
    Drive,
    Episode,
    Epoch,
    Goal,
    MemorySuggestion,
    Note,
    Playbook,
    RawEntry,
    Relationship,
    SelfNarrative,
    Summary,
    TrustAssessment,
    Value,
)
from tests.contracts.conftest import CONTRACT_STACK_ID

STACK_ID = CONTRACT_STACK_ID

# NOTE: The `stack` fixture is provided by conftest.py (parameterized by backend).


def _uid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ============================================================================
# Factory helpers — one per memory type
# ============================================================================


def _make_episode(**kw) -> Episode:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        objective="Learn Python",
        outcome="Built a web app",
        created_at=_now(),
        lessons=["Practice matters", "Read the docs"],
        tags=["programming", "python"],
    )
    defaults.update(kw)
    return Episode(**defaults)


def _make_belief(**kw) -> Belief:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        statement="Testing improves quality",
        belief_type="fact",
        confidence=0.85,
        created_at=_now(),
    )
    defaults.update(kw)
    return Belief(**defaults)


def _make_value(**kw) -> Value:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        name="Reliability",
        statement="Systems should be dependable",
        priority=75,
        created_at=_now(),
    )
    defaults.update(kw)
    return Value(**defaults)


def _make_goal(**kw) -> Goal:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        title="Ship v1.0",
        description="Release the first version",
        goal_type="task",
        priority="high",
        status="active",
        created_at=_now(),
    )
    defaults.update(kw)
    return Goal(**defaults)


def _make_note(**kw) -> Note:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        content="Important observation",
        note_type="insight",
        tags=["meta"],
        created_at=_now(),
    )
    defaults.update(kw)
    return Note(**defaults)


def _make_drive(**kw) -> Drive:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        drive_type="curiosity",
        intensity=0.7,
        focus_areas=["learning"],
        created_at=_now(),
    )
    defaults.update(kw)
    return Drive(**defaults)


def _make_relationship(**kw) -> Relationship:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        entity_name="alice",
        entity_type="human",
        relationship_type="collaborator",
        notes="Good partner",
        sentiment=0.6,
        created_at=_now(),
    )
    defaults.update(kw)
    return Relationship(**defaults)


def _make_raw(**kw) -> RawEntry:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        blob="Some brain dump text",
        captured_at=_now(),
        source="test",
    )
    defaults.update(kw)
    return RawEntry(**defaults)


def _make_playbook(**kw) -> Playbook:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        name="Deploy process",
        description="How to deploy",
        trigger_conditions=["release ready"],
        steps=[{"action": "build", "details": "run build"}],
        failure_modes=["build failure"],
        created_at=_now(),
    )
    defaults.update(kw)
    return Playbook(**defaults)


def _make_epoch(**kw) -> Epoch:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        epoch_number=1,
        name="Foundation",
        started_at=_now(),
    )
    defaults.update(kw)
    return Epoch(**defaults)


def _make_summary(**kw) -> Summary:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        scope="month",
        period_start="2025-01-01",
        period_end="2025-01-31",
        content="First month of operation",
        created_at=_now(),
    )
    defaults.update(kw)
    return Summary(**defaults)


def _make_self_narrative(**kw) -> SelfNarrative:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        content="I am a learning system",
        narrative_type="identity",
        created_at=_now(),
    )
    defaults.update(kw)
    return SelfNarrative(**defaults)


def _make_suggestion(**kw) -> MemorySuggestion:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        memory_type="belief",
        content={"statement": "Testing is good"},
        confidence=0.7,
        source_raw_ids=["raw-1"],
        created_at=_now(),
    )
    defaults.update(kw)
    return MemorySuggestion(**defaults)


def _make_trust_assessment(**kw) -> TrustAssessment:
    defaults = dict(
        id=_uid(),
        stack_id=STACK_ID,
        entity="bob",
        dimensions={"general": {"score": 0.8}},
        created_at=_now(),
        last_updated=_now(),
    )
    defaults.update(kw)
    return TrustAssessment(**defaults)


def _make_mock_component(name="test-comp", required=False, needs_inference=False):
    """Create a mock StackComponentProtocol."""
    comp = MagicMock(spec=StackComponentProtocol)
    type(comp).name = PropertyMock(return_value=name)
    type(comp).version = PropertyMock(return_value="1.0.0")
    type(comp).required = PropertyMock(return_value=required)
    type(comp).needs_inference = PropertyMock(return_value=needs_inference)
    comp.attach.return_value = None
    comp.detach.return_value = None
    comp.set_inference.return_value = None
    comp.on_save.return_value = None
    comp.on_search.return_value = []
    comp.on_load.return_value = None
    comp.on_maintenance.return_value = {"processed": 5}
    return comp


# ============================================================================
# 1. Write/Read Roundtrip Tests — each memory type
# ============================================================================


class TestEpisodeRoundtrip:
    def test_save_and_retrieve(self, stack):
        ep = _make_episode()
        returned_id = stack.save_episode(ep)
        assert returned_id == ep.id

        episodes = stack.get_episodes(limit=10)
        assert len(episodes) >= 1
        found = [e for e in episodes if e.id == ep.id]
        assert len(found) == 1
        assert found[0].objective == "Learn Python"
        assert found[0].outcome == "Built a web app"

    def test_get_by_id(self, stack):
        ep = _make_episode()
        stack.save_episode(ep)
        retrieved = stack.get_memory("episode", ep.id)
        assert retrieved is not None
        assert retrieved.id == ep.id

    def test_batch_save(self, stack):
        eps = [_make_episode(objective=f"Task {i}") for i in range(3)]
        ids = stack.save_episodes_batch(eps)
        assert len(ids) == 3
        all_eps = stack.get_episodes(limit=10)
        assert len(all_eps) >= 3


class TestBeliefRoundtrip:
    def test_save_and_retrieve(self, stack):
        b = _make_belief()
        returned_id = stack.save_belief(b)
        assert returned_id == b.id

        beliefs = stack.get_beliefs(limit=10)
        found = [x for x in beliefs if x.id == b.id]
        assert len(found) == 1
        assert found[0].statement == "Testing improves quality"
        assert found[0].confidence == 0.85

    def test_filter_by_type(self, stack):
        b1 = _make_belief(belief_type="fact")
        b2 = _make_belief(belief_type="opinion")
        stack.save_belief(b1)
        stack.save_belief(b2)

        facts = stack.get_beliefs(belief_type="fact")
        assert all(b.belief_type == "fact" for b in facts)

    def test_filter_by_min_confidence(self, stack):
        b_low = _make_belief(confidence=0.3)
        b_high = _make_belief(confidence=0.9)
        stack.save_belief(b_low)
        stack.save_belief(b_high)

        high_conf = stack.get_beliefs(min_confidence=0.8)
        assert all(b.confidence >= 0.8 for b in high_conf)

    def test_batch_save(self, stack):
        beliefs = [_make_belief(statement=f"Belief {i}") for i in range(3)]
        ids = stack.save_beliefs_batch(beliefs)
        assert len(ids) == 3


class TestValueRoundtrip:
    def test_save_and_retrieve(self, stack):
        v = _make_value()
        returned_id = stack.save_value(v)
        assert returned_id == v.id

        values = stack.get_values(limit=10)
        found = [x for x in values if x.id == v.id]
        assert len(found) == 1
        assert found[0].name == "Reliability"
        assert found[0].priority == 75


class TestGoalRoundtrip:
    def test_save_and_retrieve(self, stack):
        g = _make_goal()
        returned_id = stack.save_goal(g)
        assert returned_id == g.id

        goals = stack.get_goals(limit=10)
        found = [x for x in goals if x.id == g.id]
        assert len(found) == 1
        assert found[0].title == "Ship v1.0"
        assert found[0].status == "active"

    def test_filter_by_status(self, stack):
        g_active = _make_goal(status="active")
        g_done = _make_goal(status="completed")
        stack.save_goal(g_active)
        stack.save_goal(g_done)

        active = stack.get_goals(status="active")
        assert all(g.status == "active" for g in active)


class TestConstrainedValueValidation:
    def test_get_goals_rejects_invalid_status(self, stack):
        with pytest.raises(ValueError, match="Invalid goal status"):
            stack.get_goals(status="broken")

    def test_get_suggestions_rejects_invalid_status(self, stack):
        with pytest.raises(ValueError, match="Invalid suggestion status"):
            stack.get_suggestions(status="stalled")

    def test_get_suggestions_rejects_invalid_memory_type(self, stack):
        with pytest.raises(ValueError, match="Invalid suggestion memory_type"):
            stack.get_suggestions(memory_type="widget")

    def test_search_rejects_invalid_record_types(self, stack):
        with pytest.raises(ValueError, match="Invalid search record type"):
            stack.search("test", record_types=["episode", "invalid"])


class TestNoteRoundtrip:
    def test_save_and_retrieve(self, stack):
        n = _make_note()
        returned_id = stack.save_note(n)
        assert returned_id == n.id

        notes = stack.get_notes(limit=10)
        found = [x for x in notes if x.id == n.id]
        assert len(found) == 1
        assert found[0].content == "Important observation"

    def test_filter_by_type(self, stack):
        n1 = _make_note(note_type="insight")
        n2 = _make_note(note_type="decision")
        stack.save_note(n1)
        stack.save_note(n2)

        insights = stack.get_notes(note_type="insight")
        assert all(n.note_type == "insight" for n in insights)

    def test_batch_save(self, stack):
        notes = [_make_note(content=f"Note {i}") for i in range(3)]
        ids = stack.save_notes_batch(notes)
        assert len(ids) == 3


class TestDriveRoundtrip:
    def test_save_and_retrieve(self, stack):
        d = _make_drive()
        returned_id = stack.save_drive(d)
        assert returned_id == d.id

        drives = stack.get_drives()
        found = [x for x in drives if x.id == d.id]
        assert len(found) == 1
        assert found[0].drive_type == "curiosity"
        assert found[0].intensity == 0.7


class TestRelationshipRoundtrip:
    def test_save_and_retrieve(self, stack):
        r = _make_relationship()
        returned_id = stack.save_relationship(r)
        assert returned_id == r.id

        rels = stack.get_relationships()
        found = [x for x in rels if x.id == r.id]
        assert len(found) == 1
        assert found[0].entity_name == "alice"
        assert found[0].entity_type == "human"

    def test_filter_by_entity_id(self, stack):
        r1 = _make_relationship(entity_name="alice")
        r2 = _make_relationship(entity_name="bob")
        stack.save_relationship(r1)
        stack.save_relationship(r2)

        alice_rels = stack.get_relationships(entity_id="alice")
        assert all(r.entity_name == "alice" for r in alice_rels)

    def test_filter_by_entity_type(self, stack):
        r1 = _make_relationship(entity_type="human")
        r2 = _make_relationship(entity_type="agent")
        stack.save_relationship(r1)
        stack.save_relationship(r2)

        human_rels = stack.get_relationships(entity_type="human")
        assert all(r.entity_type == "human" for r in human_rels)


class TestRawRoundtrip:
    def test_save_and_retrieve(self, stack):
        r = _make_raw()
        returned_id = stack.save_raw(r)
        assert isinstance(returned_id, str)
        assert len(returned_id) > 0

        raw_entries = stack.get_raw(limit=10)
        assert len(raw_entries) >= 1


class TestPlaybookRoundtrip:
    def test_save_and_retrieve(self, stack):
        p = _make_playbook()
        returned_id = stack.save_playbook(p)
        assert returned_id == p.id

        # get_memory doesn't support playbook; use backend directly
        retrieved = stack._backend.get_playbook(p.id)
        assert retrieved is not None
        assert retrieved.name == "Deploy process"


class TestEpochRoundtrip:
    def test_save_and_retrieve(self, stack):
        ep = _make_epoch()
        returned_id = stack.save_epoch(ep)
        assert returned_id == ep.id

        retrieved = stack._backend.get_epoch(ep.id)
        assert retrieved is not None
        assert retrieved.name == "Foundation"


class TestSummaryRoundtrip:
    def test_save_and_retrieve(self, stack):
        s = _make_summary()
        returned_id = stack.save_summary(s)
        assert returned_id == s.id

        retrieved = stack._backend.get_summary(s.id)
        assert retrieved is not None
        assert retrieved.scope == "month"


class TestSelfNarrativeRoundtrip:
    def test_save_and_retrieve(self, stack):
        sn = _make_self_narrative()
        returned_id = stack.save_self_narrative(sn)
        assert returned_id == sn.id

        retrieved = stack._backend.get_self_narrative(sn.id)
        assert retrieved is not None
        assert retrieved.narrative_type == "identity"


class TestSuggestionRoundtrip:
    def test_save_and_retrieve(self, stack):
        s = _make_suggestion()
        returned_id = stack.save_suggestion(s)
        assert returned_id == s.id

        retrieved = stack._backend.get_suggestion(s.id)
        assert retrieved is not None
        assert retrieved.memory_type == "belief"


class TestSuggestionLifecycle:
    """Contract tests for suggestion lifecycle APIs (get, filter, accept, dismiss)."""

    def test_get_suggestion_returns_saved(self, stack):
        s = _make_suggestion()
        stack.save_suggestion(s)
        retrieved = stack.get_suggestion(s.id)
        assert retrieved is not None
        assert retrieved.id == s.id
        assert retrieved.memory_type == "belief"
        assert retrieved.status == "pending"

    def test_get_suggestions_filters_by_status(self, stack):
        s1 = _make_suggestion(status="pending")
        s2 = _make_suggestion(status="pending")
        stack.save_suggestion(s1)
        stack.save_suggestion(s2)
        # Dismiss one to change its status
        stack.dismiss_suggestion(s2.id, reason="not useful")

        pending = stack.get_suggestions(status="pending")
        assert any(s.id == s1.id for s in pending)
        assert not any(s.id == s2.id for s in pending)

        dismissed = stack.get_suggestions(status="dismissed")
        assert any(s.id == s2.id for s in dismissed)

    def test_get_suggestions_filters_by_memory_type(self, stack):
        s_belief = _make_suggestion(memory_type="belief")
        s_episode = _make_suggestion(
            memory_type="episode", content={"objective": "Test", "outcome": "Done"}
        )
        stack.save_suggestion(s_belief)
        stack.save_suggestion(s_episode)

        beliefs = stack.get_suggestions(memory_type="belief")
        assert any(s.id == s_belief.id for s in beliefs)
        assert not any(s.id == s_episode.id for s in beliefs)

    def test_get_suggestions_filters_by_min_confidence(self, stack):
        s_high = _make_suggestion(confidence=0.9)
        s_low = _make_suggestion(confidence=0.3)
        stack.save_suggestion(s_high)
        stack.save_suggestion(s_low)

        high_conf = stack.get_suggestions(min_confidence=0.7)
        assert any(s.id == s_high.id for s in high_conf)
        assert not any(s.id == s_low.id for s in high_conf)

    def test_get_suggestions_filters_by_source_raw_id(self, stack):
        s1 = _make_suggestion(source_raw_ids=["raw-123"])
        s2 = _make_suggestion(source_raw_ids=["raw-456"])
        stack.save_suggestion(s1)
        stack.save_suggestion(s2)

        results = stack.get_suggestions(source_raw_id="raw-123")
        assert any(s.id == s1.id for s in results)
        assert not any(s.id == s2.id for s in results)

    def test_accept_suggestion_creates_memory(self, stack):
        s = _make_suggestion(
            memory_type="belief",
            content={"statement": "Testing is essential", "belief_type": "fact"},
        )
        stack.save_suggestion(s)
        memory_id = stack.accept_suggestion(s.id)
        assert memory_id is not None

        # Verify the belief was created
        beliefs = stack.get_beliefs(limit=100)
        assert any(b.id == memory_id for b in beliefs)

    def test_accept_suggestion_marks_promoted(self, stack):
        s = _make_suggestion(
            memory_type="belief",
            content={"statement": "Promoted belief"},
        )
        stack.save_suggestion(s)
        stack.accept_suggestion(s.id)

        updated = stack.get_suggestion(s.id)
        assert updated is not None
        assert updated.status == "promoted"

    def test_dismiss_suggestion_marks_dismissed(self, stack):
        s = _make_suggestion()
        stack.save_suggestion(s)
        result = stack.dismiss_suggestion(s.id, reason="not relevant")
        assert result is True

        updated = stack.get_suggestion(s.id)
        assert updated is not None
        assert updated.status == "dismissed"

    def test_accept_unknown_type_raises(self, stack):
        s = _make_suggestion(memory_type="widget", content={"foo": "bar"})
        stack.save_suggestion(s)
        with pytest.raises(ValueError, match="[Uu]nsupported.*widget"):
            stack.accept_suggestion(s.id)


# ============================================================================
# 2. Search
# ============================================================================


class TestSearch:
    def test_search_returns_protocol_results(self, stack):
        stack.save_episode(_make_episode(objective="Learn Rust programming"))
        stack.save_belief(_make_belief(statement="Rust is memory safe"))

        results = stack.search("Rust")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ProtocolSearchResult)
            assert hasattr(r, "memory_type")
            assert hasattr(r, "memory_id")
            assert hasattr(r, "content")
            assert hasattr(r, "score")

    def test_search_with_limit(self, stack):
        for i in range(5):
            stack.save_note(_make_note(content=f"Note about topic {i}"))

        results = stack.search("topic", limit=2)
        assert len(results) <= 2

    def test_search_empty_stack(self, stack):
        results = stack.search("anything")
        assert results == []

    def test_search_with_min_confidence(self, stack):
        stack.save_belief(_make_belief(statement="High conf belief", confidence=0.95))
        stack.save_belief(_make_belief(statement="Low conf belief", confidence=0.2))

        results = stack.search("belief", min_confidence=0.8)
        for r in results:
            assert r.metadata.get("confidence", 1.0) >= 0.8


# ============================================================================
# 3. Load (Working Memory)
# ============================================================================


class TestLoad:
    def test_load_returns_dict(self, stack):
        stack.save_value(_make_value())
        stack.save_belief(_make_belief())
        stack.save_episode(_make_episode())

        result = stack.load(token_budget=8000)
        assert isinstance(result, dict)
        assert "values" in result
        assert "beliefs" in result
        assert "episodes" in result
        assert "_meta" in result

    def test_load_respects_budget(self, stack):
        # Add many items
        for i in range(20):
            stack.save_episode(_make_episode(objective=f"Long objective {i}" * 10))

        result_small = stack.load(token_budget=200)
        result_large = stack.load(token_budget=20000)

        # Smaller budget should produce fewer or equal items
        small_ep_count = len(result_small.get("episodes", []))
        large_ep_count = len(result_large.get("episodes", []))
        assert small_ep_count <= large_ep_count

    def test_load_empty_stack(self, stack):
        result = stack.load()
        assert isinstance(result, dict)

    def test_load_meta_tracks_budget(self, stack):
        stack.save_value(_make_value())
        result = stack.load(token_budget=5000)
        meta = result.get("_meta", {})
        assert "budget_total" in meta
        assert meta["budget_total"] == 5000


# ============================================================================
# 4. Forget / Recover / Protect
# ============================================================================


class TestMetaMemoryOps:
    def test_forget_and_recover(self, stack):
        ep = _make_episode()
        stack.save_episode(ep)

        # Forget
        result = stack.forget_memory("episode", ep.id, "no longer relevant")
        assert result is True

        # Forgotten episodes excluded by default
        episodes = stack.get_episodes()
        found = [e for e in episodes if e.id == ep.id]
        assert len(found) == 0

        # Include forgotten
        all_eps = stack.get_episodes(include_forgotten=True)
        forgotten = [e for e in all_eps if e.id == ep.id]
        assert len(forgotten) == 1
        assert forgotten[0].strength == 0.0

        # Recover
        recovered = stack.recover_memory("episode", ep.id)
        assert recovered is True

        episodes_after = stack.get_episodes(include_weak=True)
        found_after = [e for e in episodes_after if e.id == ep.id]
        assert len(found_after) == 1
        assert found_after[0].strength > 0.0

    def test_protect_memory(self, stack):
        b = _make_belief()
        stack.save_belief(b)

        result = stack.protect_memory("belief", b.id, True)
        assert result is True

        beliefs = stack.get_beliefs()
        found = [x for x in beliefs if x.id == b.id]
        assert len(found) == 1
        assert found[0].is_protected is True

        # Unprotect
        stack.protect_memory("belief", b.id, False)
        beliefs2 = stack.get_beliefs()
        found2 = [x for x in beliefs2 if x.id == b.id]
        assert found2[0].is_protected is False

    def test_record_access(self, stack):
        ep = _make_episode()
        stack.save_episode(ep)

        result = stack.record_access("episode", ep.id)
        assert result is True

    def test_update_memory_meta_confidence(self, stack):
        b = _make_belief(confidence=0.5)
        stack.save_belief(b)

        result = stack.update_memory_meta("belief", b.id, confidence=0.95)
        assert result is True

        beliefs = stack.get_beliefs()
        found = [x for x in beliefs if x.id == b.id]
        assert found[0].confidence == pytest.approx(0.95, abs=0.01)


# ============================================================================
# 5. Stats
# ============================================================================


class TestStats:
    def test_get_stats_empty(self, stack):
        stats = stack.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_with_data(self, stack):
        stack.save_episode(_make_episode())
        stack.save_belief(_make_belief())
        stack.save_value(_make_value())

        stats = stack.get_stats()
        assert stats.get("episodes", 0) >= 1
        assert stats.get("beliefs", 0) >= 1
        assert stats.get("values", 0) >= 1


# ============================================================================
# 6. Composition Hooks
# ============================================================================


class TestCompositionHooks:
    def test_on_attach_tracks_core_id(self, stack):
        stack.on_attach("core-123")
        assert stack._attached_core_id == "core-123"

    def test_on_detach_clears_state(self, stack):
        stack.on_attach("core-123")
        stack.on_detach("core-123")
        assert stack._attached_core_id is None
        assert stack._inference is None

    def test_on_attach_with_inference(self, stack):
        inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-123", inference=inference)
        assert stack._inference is inference

    def test_on_model_changed_updates_inference(self, stack):
        inference1 = MagicMock(spec=InferenceService)
        inference2 = MagicMock(spec=InferenceService)

        stack.on_attach("core-123", inference=inference1)
        assert stack._inference is inference1

        stack.on_model_changed(inference2)
        assert stack._inference is inference2

    def test_on_model_changed_none_clears(self, stack):
        inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-123", inference=inference)
        stack.on_model_changed(None)
        assert stack._inference is None

    def test_hooks_propagate_to_components(self, stack):
        comp = _make_mock_component("test-comp")
        stack.add_component(comp)

        inference = MagicMock(spec=InferenceService)
        stack.on_attach("core-123", inference=inference)
        comp.set_inference.assert_called_with(inference)

        stack.on_model_changed(None)
        comp.set_inference.assert_called_with(None)

        stack.on_detach("core-123")
        assert comp.set_inference.call_count >= 2


# ============================================================================
# 7. Detached Operation (No Core)
# ============================================================================


class TestDetachedOperation:
    """Stack works independently without any core attached."""

    def test_write_and_read_detached(self, stack):
        ep = _make_episode()
        stack.save_episode(ep)
        episodes = stack.get_episodes()
        assert len(episodes) >= 1

    def test_search_detached(self, stack):
        stack.save_note(_make_note(content="detached search test"))
        results = stack.search("detached")
        assert isinstance(results, list)

    def test_load_detached(self, stack):
        stack.save_value(_make_value())
        result = stack.load()
        assert isinstance(result, dict)

    def test_stats_detached(self, stack):
        stats = stack.get_stats()
        assert isinstance(stats, dict)


# ============================================================================
# 8. Component Registry
# ============================================================================


class TestComponentRegistry:
    def test_add_component(self, stack):
        comp = _make_mock_component("embedding")
        stack.add_component(comp)

        assert "embedding" in stack.components
        comp.attach.assert_called_once_with(STACK_ID, None)

    def test_add_component_duplicate_raises(self, stack):
        comp = _make_mock_component("embedding")
        stack.add_component(comp)

        with pytest.raises(ValueError, match="already registered"):
            stack.add_component(comp)

    def test_remove_component(self, stack):
        comp = _make_mock_component("optional")
        stack.add_component(comp)

        stack.remove_component("optional")
        assert "optional" not in stack.components
        comp.detach.assert_called_once()

    def test_remove_required_raises(self, stack):
        comp = _make_mock_component("critical", required=True)
        stack.add_component(comp)

        with pytest.raises(ValueError, match="Cannot remove required"):
            stack.remove_component("critical")

    def test_remove_missing_raises(self, stack):
        with pytest.raises(ValueError, match="not found"):
            stack.remove_component("nonexistent")

    def test_get_component(self, stack):
        comp = _make_mock_component("test-comp")
        stack.add_component(comp)

        result = stack.get_component("test-comp")
        assert result is not None
        assert result.name == "test-comp"

    def test_get_component_missing_returns_none(self, stack):
        assert stack.get_component("missing") is None

    def test_maintenance_runs_all_components(self, stack):
        comp1 = _make_mock_component("comp1")
        comp2 = _make_mock_component("comp2")
        stack.add_component(comp1)
        stack.add_component(comp2)

        results = stack.maintenance()
        assert "comp1" in results
        assert "comp2" in results


# ============================================================================
# 9. Trust Layer
# ============================================================================


class TestTrustLayer:
    def test_save_and_get_trust_assessment(self, stack):
        ta = _make_trust_assessment(entity="partner-1")
        stack.save_trust_assessment(ta)

        assessments = stack.get_trust_assessments(entity_id="partner-1")
        assert len(assessments) >= 1
        assert assessments[0].entity == "partner-1"

    def test_compute_trust_known_entity(self, stack):
        ta = _make_trust_assessment(
            entity="trusted-one",
            dimensions={"general": {"score": 0.9}},
        )
        stack.save_trust_assessment(ta)

        result = stack.compute_trust("trusted-one")
        assert result["score"] == pytest.approx(0.9)

    def test_compute_trust_unknown_entity(self, stack):
        result = stack.compute_trust("stranger")
        assert result["score"] == pytest.approx(0.5)
        assert result["source"] == "default"


# ============================================================================
# 10. Properties
# ============================================================================


class TestStackProperties:
    def test_stack_id(self, stack):
        assert stack.stack_id == STACK_ID

    def test_schema_version_is_int(self, stack):
        assert isinstance(stack.schema_version, int)
        assert stack.schema_version > 0

    def test_components_initially_empty(self, stack):
        assert stack.components == {}


# ============================================================================
# 11. Export
# ============================================================================


class TestExport:
    def test_dump_markdown(self, stack):
        stack.save_episode(_make_episode())
        stack.save_belief(_make_belief())

        md = stack.dump(format="markdown")
        assert isinstance(md, str)
        assert "# Memory Dump" in md

    def test_dump_json(self, stack):
        stack.save_value(_make_value())

        json_str = stack.dump(format="json")
        data = json.loads(json_str)
        assert "stack_id" in data
        assert "values" in data

    def test_export_to_file(self, stack, tmp_path):
        stack.save_note(_make_note())

        export_path = str(tmp_path / "export.md")
        stack.export(export_path)
        content = (tmp_path / "export.md").read_text()
        assert len(content) > 0

    def test_dump_rejects_invalid_format(self, stack):
        with pytest.raises(ValueError, match="Invalid dump format"):
            stack.dump(format="xml")

    def test_export_rejects_invalid_format(self, stack, tmp_path):
        with pytest.raises(ValueError, match="Invalid export format"):
            stack.export(str(tmp_path / "export.bin"), format="binary")


class TestProcessingConfig:
    def test_set_processing_config_rejects_invalid_transition(self, stack):
        with pytest.raises(ValueError, match="Invalid processing transition"):
            stack.set_processing_config("bogus", enabled=True)


# ============================================================================
# 12. Features (consolidate, apply_forgetting)
# ============================================================================


class TestFeatures:
    def test_consolidate_needs_episodes(self, stack):
        result = stack.consolidate()
        assert result["consolidated"] == 0

    def test_consolidate_with_episodes(self, stack):
        for i in range(5):
            stack.save_episode(
                _make_episode(
                    objective=f"Task {i}",
                    lessons=["Be thorough", "Test first"],
                )
            )

        result = stack.consolidate()
        assert result["consolidated"] >= 3
        assert "lessons_found" in result

    def test_apply_forgetting(self, stack):
        stack.save_episode(_make_episode())
        result = stack.apply_forgetting()
        assert isinstance(result, dict)
        assert "forgotten" in result


# ============================================================================
# 13. Sync Stubs (local mode)
# ============================================================================


class TestSyncLocal:
    def test_sync_returns_result(self, stack):
        result = stack.sync()
        assert hasattr(result, "pushed")
        assert hasattr(result, "pulled")

    def test_is_online(self, stack):
        assert isinstance(stack.is_online(), bool)

    def test_pending_sync_count(self, stack):
        count = stack.get_pending_sync_count()
        assert isinstance(count, int)
        assert count >= 0


# ============================================================================
# 14. Component Hook Contracts
# ============================================================================


class TestComponentHookContract:
    """Contract: StackProtocol implementations MUST dispatch component hooks."""

    def _make_tracking_component(self):
        comp = MagicMock(spec=StackComponentProtocol)
        comp.name = "hook-tracker"
        comp.required = False
        comp.needs_inference = False
        comp.on_save = MagicMock()
        comp.on_search = MagicMock(return_value=None)
        comp.on_load = MagicMock()
        comp.on_maintenance = MagicMock(return_value=None)
        comp.on_model_changed = MagicMock()
        comp.attach = MagicMock()
        comp.detach = MagicMock()
        return comp

    def test_save_dispatches_on_save(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        ep = _make_episode()
        eid = stack.save_episode(ep)
        comp.on_save.assert_called_once_with("episode", eid, ep)

    def test_search_dispatches_on_search(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        stack.save_episode(_make_episode(objective="hook test"))
        stack.search("hook test")
        comp.on_search.assert_called_once()

    def test_load_dispatches_on_load(self, stack):
        comp = self._make_tracking_component()
        stack.add_component(comp)
        stack.load()
        comp.on_load.assert_called_once()
