"""Tests for kernle/cli/commands/relations.py — relation and entity model commands."""

import argparse
import json

import pytest

from kernle import Kernle
from kernle.cli.commands.relations import cmd_entity_model, cmd_relation
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model


@pytest.fixture
def k(tmp_path):
    storage = SQLiteStorage(stack_id="test-rel", db_path=tmp_path / "rel.db")
    inst = Kernle(stack_id="test-rel", storage=storage, strict=False)
    bind_noop_model(inst)
    yield inst
    storage.close()


def _rel_args(**kwargs):
    defaults = {"command": "relation", "relation_action": "list"}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _em_args(**kwargs):
    defaults = {
        "command": "entity-model",
        "entity_model_action": "list",
        "limit": 100,
        "json": False,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# ──────────────────────────────────────────────
# cmd_relation — list
# ──────────────────────────────────────────────


class TestRelationList:
    def test_list_empty(self, k, capsys):
        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        assert "No relationships recorded yet." in out
        assert "kernle relation add" in out

    def test_list_shows_added_relationships(self, k, capsys):
        k.relationship("Alice", trust_level=0.8, entity_type="person", notes="Friend")
        k.relationship("BotX", trust_level=0.5, entity_type="si")

        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        assert "Alice (person)" in out
        assert "BotX (si)" in out
        assert "Trust:" in out

    def test_list_trust_bar_rendering(self, k, capsys):
        # trust_level=1.0 -> sentiment=1.0 -> pct=100
        k.relationship("HighTrust", trust_level=1.0, entity_type="person")
        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        # 100% -> 10 filled blocks, 0 empty
        assert "██████████" in out
        assert "100%" in out

    def test_list_low_trust_bar(self, k, capsys):
        # trust_level=0.0 -> sentiment=-1.0 -> pct=0
        k.relationship("LowTrust", trust_level=0.0, entity_type="person")
        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        assert "░░░░░░░░░░" in out
        assert "0%" in out

    def test_list_truncates_long_notes(self, k, capsys):
        long_notes = "A" * 100
        k.relationship("Verbose", trust_level=0.5, notes=long_notes)
        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        # Notes should be truncated at 60 chars + "..."
        assert "A" * 60 + "..." in out

    def test_list_shows_interaction_count(self, k, capsys):
        k.relationship("Bob", trust_level=0.5)
        k.relationship("Bob", interaction_type="chat")
        cmd_relation(_rel_args(relation_action="list"), k)
        out = capsys.readouterr().out
        assert "Interactions: 2" in out


# ──────────────────────────────────────────────
# cmd_relation — add
# ──────────────────────────────────────────────


class TestRelationAdd:
    def test_add_basic(self, k, capsys):
        args = _rel_args(
            relation_action="add",
            name="Alice",
            type="person",
            trust=0.7,
            notes="Colleague",
            derived_from=None,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Relationship added: Alice" in out
        assert "Type: person" in out
        assert "Trust: 70%" in out

        # Verify persisted
        rels = k.load_relationships(limit=50)
        assert len(rels) == 1
        assert rels[0]["entity_name"] == "Alice"
        assert rels[0]["entity_type"] == "person"

    def test_add_defaults(self, k, capsys):
        """type defaults to 'person', trust defaults to 0.5."""
        args = _rel_args(
            relation_action="add",
            name="DefaultEntity",
            type=None,
            trust=None,
            notes=None,
            derived_from=None,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Type: person" in out
        assert "Trust: 50%" in out

    def test_add_with_derived_from(self, k, capsys):
        args = _rel_args(
            relation_action="add",
            name="Charlie",
            type="si",
            trust=0.9,
            notes=None,
            derived_from=["mem-1", "mem-2"],
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Derived from: 2 memories" in out

    def test_add_entity_types(self, k, capsys):
        for etype in ("person", "si", "org", "system"):
            args = _rel_args(
                relation_action="add",
                name=f"Entity-{etype}",
                type=etype,
                trust=0.5,
                notes=None,
                derived_from=None,
            )
            cmd_relation(args, k)

        rels = k.load_relationships(limit=50)
        types_found = {r["entity_type"] for r in rels}
        assert types_found == {"person", "si", "org", "system"}


# ──────────────────────────────────────────────
# cmd_relation — update
# ──────────────────────────────────────────────


class TestRelationUpdate:
    def test_update_trust(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person")
        args = _rel_args(
            relation_action="update",
            name="Alice",
            trust=0.9,
            notes=None,
            type=None,
            derived_from=None,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Relationship updated: Alice" in out

        rels = k.load_relationships(limit=50)
        alice = next(r for r in rels if r["entity_name"] == "Alice")
        # trust_level=0.9 -> sentiment=0.8
        assert alice["sentiment"] == pytest.approx(0.8, abs=0.01)

    def test_update_notes(self, k, capsys):
        k.relationship("Bob", trust_level=0.5, notes="Original")
        args = _rel_args(
            relation_action="update",
            name="Bob",
            trust=None,
            notes="Updated notes",
            type=None,
            derived_from=None,
        )
        cmd_relation(args, k)

        rels = k.load_relationships(limit=50)
        bob = next(r for r in rels if r["entity_name"] == "Bob")
        assert bob["notes"] == "Updated notes"

    def test_update_no_fields_prints_error(self, k, capsys):
        args = _rel_args(
            relation_action="update",
            name="Nobody",
            trust=None,
            notes=None,
            type=None,
            derived_from=None,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Provide --trust, --notes, --type, or --derived-from to update" in out

    def test_update_with_derived_from(self, k, capsys):
        k.relationship("Alice", trust_level=0.5)
        args = _rel_args(
            relation_action="update",
            name="Alice",
            trust=0.7,
            notes=None,
            type=None,
            derived_from=["mem-x"],
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Derived from: 1 memories" in out


# ──────────────────────────────────────────────
# cmd_relation — show
# ──────────────────────────────────────────────


class TestRelationShow:
    def test_show_existing(self, k, capsys):
        # trust_level=0.8 -> sentiment = 0.6 -> pct = int(((0.6+1)/2)*100) = 80
        k.relationship("Alice", trust_level=0.8, entity_type="person", notes="Good friend")
        args = _rel_args(relation_action="show", name="Alice")
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "## Alice" in out
        assert "Type: person" in out
        assert "Trust: 80%" in out
        assert "Notes:\nGood friend" in out

    def test_show_case_insensitive(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person")
        args = _rel_args(relation_action="show", name="alice")
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "## Alice" in out

    def test_show_not_found(self, k, capsys):
        args = _rel_args(relation_action="show", name="Nobody")
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "No relationship found for 'Nobody'" in out

    def test_show_interaction_count(self, k, capsys):
        k.relationship("Bob", trust_level=0.5)
        k.relationship("Bob", interaction_type="call")
        k.relationship("Bob", interaction_type="email")
        args = _rel_args(relation_action="show", name="Bob")
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Interactions: 3" in out


# ──────────────────────────────────────────────
# cmd_relation — log
# ──────────────────────────────────────────────


class TestRelationLog:
    def test_log_interaction(self, k, capsys):
        k.relationship("Alice", trust_level=0.5)
        args = _rel_args(relation_action="log", name="Alice", interaction="meeting")
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Logged interaction with Alice: meeting" in out

    def test_log_default_interaction(self, k, capsys):
        k.relationship("Alice", trust_level=0.5)
        args = _rel_args(relation_action="log", name="Alice", interaction=None)
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "Logged interaction with Alice: interaction" in out

    def test_log_increments_count(self, k, capsys):
        k.relationship("Alice", trust_level=0.5)
        cmd_relation(_rel_args(relation_action="log", name="Alice", interaction="chat"), k)
        cmd_relation(_rel_args(relation_action="log", name="Alice", interaction="call"), k)

        rels = k.load_relationships(limit=50)
        alice = next(r for r in rels if r["entity_name"] == "Alice")
        # Initial add = 1, two logs = +2 = 3
        assert alice["interaction_count"] == 3


# ──────────────────────────────────────────────
# cmd_relation — history
# ──────────────────────────────────────────────


class TestRelationHistory:
    def test_history_empty(self, k, capsys):
        args = _rel_args(relation_action="history", name="Nobody", type=None, limit=50, json=False)
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "No history found for 'Nobody'" in out

    def test_history_shows_entries(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person")
        k.relationship("Alice", trust_level=0.9)  # triggers trust_change history

        args = _rel_args(relation_action="history", name="Alice", type=None, limit=50, json=False)
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "History for Alice" in out
        assert "entries):" in out

    def test_history_json_output(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person")
        k.relationship("Alice", trust_level=0.9)

        args = _rel_args(relation_action="history", name="Alice", type=None, limit=50, json=True)
        cmd_relation(args, k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert isinstance(parsed, list)
        assert len(parsed) >= 1

    def test_history_event_icons(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person", notes="Initial")
        k.relationship("Alice", trust_level=0.9, notes="Updated")

        args = _rel_args(relation_action="history", name="Alice", type=None, limit=50, json=False)
        cmd_relation(args, k)
        out = capsys.readouterr().out
        # trust_change -> "~~", note -> "##"
        assert "~~" in out or ">>" in out  # at least one event icon present

    def test_history_with_event_type_filter(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, entity_type="person", notes="v1")
        k.relationship("Alice", trust_level=0.9, notes="v2")

        args = _rel_args(
            relation_action="history",
            name="Alice",
            type="trust_change",
            limit=50,
            json=True,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        for entry in parsed:
            assert entry["event_type"] == "trust_change"

    def test_history_shows_old_new_values(self, k, capsys):
        k.relationship("Alice", trust_level=0.5, notes="First note")
        k.relationship("Alice", notes="Second note")

        args = _rel_args(
            relation_action="history",
            name="Alice",
            type="note",
            limit=50,
            json=False,
        )
        cmd_relation(args, k)
        out = capsys.readouterr().out
        assert "From:" in out
        assert "To:" in out


# ──────────────────────────────────────────────
# cmd_entity_model — add
# ──────────────────────────────────────────────


class TestEntityModelAdd:
    def test_add_basic(self, k, capsys):
        args = _em_args(
            entity_model_action="add",
            entity="Alice",
            observation="Prefers email over Slack",
            type="preference",
            confidence=0.85,
            episode=None,
        )
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "Entity model added:" in out
        assert "Entity: Alice" in out
        assert "Type: preference" in out
        assert "Prefers email over Slack" in out

        # Verify persisted
        models = k.get_entity_models(entity_name="Alice")
        assert len(models) == 1
        assert models[0]["model_type"] == "preference"
        assert models[0]["confidence"] == 0.85

    def test_add_truncates_long_observation_in_output(self, k, capsys):
        long_obs = "X" * 100
        args = _em_args(
            entity_model_action="add",
            entity="Bob",
            observation=long_obs,
            type="behavioral",
            confidence=0.7,
            episode=None,
        )
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "X" * 60 + "..." in out

    def test_add_with_source_episodes(self, k, capsys):
        args = _em_args(
            entity_model_action="add",
            entity="Charlie",
            observation="Good at Python",
            type="capability",
            confidence=0.9,
            episode=["ep-1", "ep-2"],
        )
        cmd_entity_model(args, k)

        models = k.get_entity_models(entity_name="Charlie")
        assert len(models) == 1
        assert models[0]["source_episodes"] == ["ep-1", "ep-2"]

    def test_add_all_model_types(self, k, capsys):
        for mtype in ("behavioral", "preference", "capability"):
            args = _em_args(
                entity_model_action="add",
                entity="Alice",
                observation=f"Observation for {mtype}",
                type=mtype,
                confidence=0.7,
                episode=None,
            )
            cmd_entity_model(args, k)

        models = k.get_entity_models(entity_name="Alice")
        assert len(models) == 3
        types = {m["model_type"] for m in models}
        assert types == {"behavioral", "preference", "capability"}


# ──────────────────────────────────────────────
# cmd_entity_model — list
# ──────────────────────────────────────────────


class TestEntityModelList:
    def test_list_empty(self, k, capsys):
        cmd_entity_model(_em_args(entity_model_action="list"), k)
        out = capsys.readouterr().out
        assert "No entity models found." in out

    def test_list_shows_models(self, k, capsys):
        k.add_entity_model("Alice", "behavioral", "Arrives early", confidence=0.8)
        k.add_entity_model("Bob", "preference", "Likes dark mode", confidence=0.6)

        cmd_entity_model(_em_args(entity_model_action="list"), k)
        out = capsys.readouterr().out
        assert "Entity Models (2 total)" in out
        assert "[B] Alice" in out
        assert "[P] Bob" in out
        assert "80%" in out
        assert "60%" in out

    def test_list_json_output(self, k, capsys):
        k.add_entity_model("Alice", "behavioral", "Arrives early", confidence=0.8)

        cmd_entity_model(_em_args(entity_model_action="list", json=True), k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert len(parsed) == 1
        assert parsed[0]["entity_name"] == "Alice"

    def test_list_filter_by_entity(self, k, capsys):
        k.add_entity_model("Alice", "behavioral", "Obs A", confidence=0.8)
        k.add_entity_model("Bob", "preference", "Obs B", confidence=0.6)

        cmd_entity_model(_em_args(entity_model_action="list", entity="Alice"), k)
        out = capsys.readouterr().out
        assert "Alice" in out
        assert "Bob" not in out

    def test_list_filter_by_type(self, k, capsys):
        k.add_entity_model("Alice", "behavioral", "Obs A", confidence=0.8)
        k.add_entity_model("Alice", "preference", "Obs B", confidence=0.6)

        cmd_entity_model(_em_args(entity_model_action="list", type="behavioral"), k)
        out = capsys.readouterr().out
        assert "[B]" in out
        assert "[P]" not in out

    def test_list_truncates_long_observations(self, k, capsys):
        long_obs = "Z" * 80
        k.add_entity_model("Alice", "behavioral", long_obs, confidence=0.7)

        cmd_entity_model(_em_args(entity_model_action="list"), k)
        out = capsys.readouterr().out
        assert "Z" * 50 + "..." in out

    def test_list_type_icons(self, k, capsys):
        k.add_entity_model("Alice", "behavioral", "obs1", confidence=0.7)
        k.add_entity_model("Bob", "preference", "obs2", confidence=0.7)
        k.add_entity_model("Charlie", "capability", "obs3", confidence=0.7)

        cmd_entity_model(_em_args(entity_model_action="list"), k)
        out = capsys.readouterr().out
        assert "[B]" in out
        assert "[P]" in out
        assert "[C]" in out


# ──────────────────────────────────────────────
# cmd_entity_model — show
# ──────────────────────────────────────────────


class TestEntityModelShow:
    def test_show_existing(self, k, capsys):
        mid = k.add_entity_model("Alice", "behavioral", "Tends to be direct", confidence=0.85)
        args = _em_args(entity_model_action="show", id=mid)
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "[B] Entity Model: Alice" in out
        assert f"ID: {mid}" in out
        assert "Type: behavioral" in out
        assert "Confidence: 85%" in out
        assert "Observation:\n  Tends to be direct" in out

    def test_show_not_found(self, k, capsys):
        args = _em_args(entity_model_action="show", id="nonexistent-id")
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "Entity model nonexistent-id not found." in out

    def test_show_json_output(self, k, capsys):
        mid = k.add_entity_model("Alice", "preference", "Likes tea", confidence=0.9)
        args = _em_args(entity_model_action="show", id=mid, json=True)
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["id"] == mid
        assert parsed["entity_name"] == "Alice"
        assert parsed["model_type"] == "preference"

    def test_show_with_source_episodes(self, k, capsys):
        mid = k.add_entity_model(
            "Alice",
            "capability",
            "Good at debugging",
            confidence=0.9,
            source_episodes=["ep-100", "ep-200"],
        )
        args = _em_args(entity_model_action="show", id=mid)
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "Source Episodes: ep-100, ep-200" in out

    def test_show_capability_type_icon(self, k, capsys):
        mid = k.add_entity_model("Alice", "capability", "Fast learner", confidence=0.5)
        args = _em_args(entity_model_action="show", id=mid)
        cmd_entity_model(args, k)
        out = capsys.readouterr().out
        assert "[C] Entity Model: Alice" in out
