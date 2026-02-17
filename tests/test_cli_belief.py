"""Tests for CLI belief command module."""

from argparse import Namespace
from unittest.mock import MagicMock

from kernle.cli.commands.belief import cmd_belief


class TestCmdBeliefRevise:
    """Test belief revise command."""

    def test_revise_success(self, capsys):
        """Successful belief revision."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [{"statement": "Test works", "belief_id": "b123"}],
            "contradicted": [],
            "suggested_new": [],
        }

        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=False,
        )

        cmd_belief(args, k)

        k.revise_beliefs_from_episode.assert_called_with("ep123")
        captured = capsys.readouterr()
        assert "Reinforced" in captured.out
        assert "Test works" in captured.out

    def test_revise_error(self, capsys):
        """Revision with error."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "error": "Episode not found",
        }

        args = Namespace(
            belief_action="revise",
            episode_id="nonexistent",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out

    def test_revise_json(self, capsys):
        """Revise JSON output."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [{"statement": "Test", "belief_id": "b123"}],
            "contradicted": [],
            "suggested_new": [],
        }

        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=True,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert '"reinforced"' in captured.out

    def test_revise_with_contradictions(self, capsys):
        """Revision with contradictions."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [],
            "contradicted": [
                {
                    "statement": "Contradicting statement here with enough content to display",
                    "belief_id": "b456",
                    "evidence": "Evidence for the contradiction here",
                }
            ],
            "suggested_new": [],
        }

        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Potential Contradictions" in captured.out
        assert "Contradicting statement" in captured.out

    def test_revise_with_suggested(self, capsys):
        """Revision with suggested new beliefs."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [],
            "contradicted": [],
            "suggested_new": [
                {
                    "statement": "New suggested belief statement here with content",
                    "suggested_confidence": 0.75,
                }
            ],
        }

        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Suggested New Beliefs" in captured.out
        assert "75%" in captured.out

    def test_revise_no_changes(self, capsys):
        """Revision with no changes found."""
        k = MagicMock()
        k.revise_beliefs_from_episode.return_value = {
            "reinforced": [],
            "contradicted": [],
            "suggested_new": [],
        }

        args = Namespace(
            belief_action="revise",
            episode_id="ep123",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "No belief revisions found" in captured.out


class TestCmdBeliefContradictions:
    """Test belief contradictions command."""

    def test_no_contradictions(self, capsys):
        """No contradictions found."""
        k = MagicMock()
        k.find_contradictions.return_value = []

        args = Namespace(
            belief_action="contradictions",
            statement="The sky is blue",
            limit=10,
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "No contradictions found" in captured.out

    def test_with_contradictions(self, capsys):
        """Contradictions found."""
        k = MagicMock()
        k.find_contradictions.return_value = [
            {
                "belief_id": "b123",
                "statement": "Contradicting belief",
                "contradiction_confidence": 0.8,
                "contradiction_type": "semantic",
                "is_active": True,
                "times_reinforced": 2,
                "explanation": "Direct conflict",
            }
        ]

        args = Namespace(
            belief_action="contradictions",
            statement="Test statement",
            limit=10,
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Potential Contradictions" in captured.out
        assert "Contradicting belief" in captured.out

    def test_contradictions_json(self, capsys):
        """Contradictions JSON output."""
        k = MagicMock()
        k.find_contradictions.return_value = [
            {"belief_id": "b123", "statement": "test", "contradiction_confidence": 0.7}
        ]

        args = Namespace(
            belief_action="contradictions",
            statement="Test",
            limit=10,
            json=True,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert '"belief_id"' in captured.out


class TestCmdBeliefHistory:
    """Test belief history command."""

    def test_history_not_found(self, capsys):
        """History for non-existent belief."""
        k = MagicMock()
        k.get_belief_history.return_value = []

        args = Namespace(
            belief_action="history",
            id="nonexistent",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "No history found" in captured.out

    def test_history_found(self, capsys):
        """History for existing belief."""
        k = MagicMock()
        k.get_belief_history.return_value = [
            {
                "id": "b123",
                "statement": "Original belief",
                "confidence": 0.7,
                "is_current": False,
                "is_active": False,
                "times_reinforced": 1,
                "created_at": "2026-01-01",
                "supersession_reason": "Updated understanding",
                "superseded_by": "b456",
            },
            {
                "id": "b456",
                "statement": "Updated belief",
                "confidence": 0.9,
                "is_current": True,
                "is_active": True,
                "times_reinforced": 3,
                "created_at": "2026-01-15",
                "supersession_reason": None,
                "superseded_by": None,
            },
        ]

        args = Namespace(
            belief_action="history",
            id="b123",
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Belief Revision History" in captured.out
        assert "Original belief" in captured.out
        assert "Updated belief" in captured.out

    def test_history_json(self, capsys):
        """History JSON output."""
        k = MagicMock()
        k.get_belief_history.return_value = [{"id": "b123", "statement": "test", "confidence": 0.8}]

        args = Namespace(
            belief_action="history",
            id="b123",
            json=True,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out
        assert '"b123"' in captured.out


class TestCmdBeliefReinforce:
    """Test belief reinforce command."""

    def test_reinforce_success(self, capsys):
        """Successful reinforcement."""
        k = MagicMock()
        k.reinforce_belief.return_value = True

        args = Namespace(
            belief_action="reinforce",
            id="b123",
        )

        cmd_belief(args, k)

        k.reinforce_belief.assert_called_with("b123", evidence_source=None, reason=None)
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "reinforced" in captured.out

    def test_reinforce_not_found(self, capsys):
        """Reinforcement of non-existent belief."""
        k = MagicMock()
        k.reinforce_belief.return_value = False

        args = Namespace(
            belief_action="reinforce",
            id="nonexistent",
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "not found" in captured.out


class TestCmdBeliefSupersede:
    """Test belief supersede command."""

    def test_supersede_success(self, capsys):
        """Successful supersession."""
        k = MagicMock()
        k.revise_belief.return_value = "new-b456"

        args = Namespace(
            belief_action="supersede",
            old_id="b123",
            new_statement="New understanding",
            confidence=0.85,
            reason="Better evidence",
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "revised" in captured.out
        assert "new-b456" in captured.out

    def test_supersede_error(self, capsys):
        """Supersession with error."""
        k = MagicMock()
        k.revise_belief.side_effect = ValueError("Belief not found")

        args = Namespace(
            belief_action="supersede",
            old_id="nonexistent",
            new_statement="New statement",
            confidence=0.8,
            reason="Test",
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "✗" in captured.out


class TestCmdBeliefList:
    """Test belief list command."""

    def test_list_empty(self, capsys):
        """Empty belief list."""
        k = MagicMock()
        k._storage.get_beliefs.return_value = []

        args = Namespace(
            belief_action="list",
            limit=20,
            all=False,
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Beliefs" in captured.out
        assert "0 total" in captured.out

    def test_list_with_beliefs(self, capsys):
        """List with beliefs."""
        belief1 = MagicMock()
        belief1.id = "b123"
        belief1.statement = "Test belief"
        belief1.confidence = 0.8
        belief1.times_reinforced = 2
        belief1.is_active = True
        belief1.supersedes = None
        belief1.superseded_by = None
        belief1.created_at = MagicMock()
        belief1.created_at.isoformat.return_value = "2026-01-01T00:00:00Z"

        k = MagicMock()
        k._storage.get_beliefs.return_value = [belief1]

        args = Namespace(
            belief_action="list",
            limit=20,
            all=False,
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Test belief" in captured.out
        assert "80%" in captured.out

    def test_list_with_supersession_chain(self, capsys):
        """List with beliefs that have supersession chain."""
        belief1 = MagicMock()
        belief1.id = "b123"
        belief1.statement = "Old belief that was superseded"
        belief1.confidence = 0.6
        belief1.times_reinforced = 0
        belief1.is_active = False
        belief1.supersedes = None
        belief1.superseded_by = "b456"
        belief1.created_at = MagicMock()
        belief1.created_at.isoformat.return_value = "2026-01-01T00:00:00Z"

        belief2 = MagicMock()
        belief2.id = "b456"
        belief2.statement = "New belief that supersedes old"
        belief2.confidence = 0.9
        belief2.times_reinforced = 3
        belief2.is_active = True
        belief2.supersedes = "b123"
        belief2.superseded_by = None
        belief2.created_at = MagicMock()
        belief2.created_at.isoformat.return_value = "2026-01-15T00:00:00Z"

        k = MagicMock()
        k._storage.get_beliefs.return_value = [belief1, belief2]

        args = Namespace(
            belief_action="list",
            limit=20,
            all=True,
            json=False,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert "Supersedes:" in captured.out
        assert "Superseded by:" in captured.out
        assert "🟢" in captured.out  # Active
        assert "⚫" in captured.out  # Inactive

    def test_list_json(self, capsys):
        """List JSON output."""
        belief1 = MagicMock()
        belief1.id = "b123"
        belief1.statement = "Test belief"
        belief1.confidence = 0.8
        belief1.times_reinforced = 2
        belief1.is_active = True
        belief1.supersedes = None
        belief1.superseded_by = None
        belief1.created_at = MagicMock()
        belief1.created_at.isoformat.return_value = "2026-01-01T00:00:00Z"

        k = MagicMock()
        k._storage.get_beliefs.return_value = [belief1]

        args = Namespace(
            belief_action="list",
            limit=20,
            all=False,
            json=True,
        )

        cmd_belief(args, k)

        captured = capsys.readouterr()
        assert '"id"' in captured.out
        assert '"statement"' in captured.out
        assert '"confidence"' in captured.out
