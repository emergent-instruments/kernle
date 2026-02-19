"""Tests for memory lint — reject malformed or low-signal beliefs/values.

Tests cover:
- lint_text: core lint function with all rule types
- lint_belief: belief-specific linting
- lint_value: value-specific linting (name + statement)
- Configuration: custom configs, disabled lint, settings-based config
- Integration: lint in save_belief/save_value redirects to suggestions
- Audit log: lint failures are surfaced in audit entries
- Known-bad patterns: truncated, templated, fragment, prefix artifacts
- Known-good patterns: valid beliefs and values pass lint
"""

import json
import uuid
from datetime import datetime, timezone

import pytest

from kernle.lint import (
    DEFAULT_LINT_CONFIG,
    LintResult,
    get_lint_config,
    lint_belief,
    lint_text,
    lint_value,
)
from kernle.stack import Stack
from kernle.types import Belief, Episode, Value

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def tmp_db(tmp_path):
    return tmp_path / "test_lint.db"


@pytest.fixture
def stack(tmp_db):
    """Stack with lint enabled (default), provenance OFF for simpler tests.

    Transitions to ACTIVE state so lint is enforced (lint only runs in ACTIVE).
    """
    s = Stack.from_sqlite(
        stack_id="test-lint",
        db_path=tmp_db,
        components=[],
        enforce_provenance=False,
    )
    # Transition to ACTIVE so lint runs
    s.on_attach("test-core")
    return s


@pytest.fixture
def stack_with_provenance(tmp_db):
    """Stack with both lint and provenance enabled."""
    return Stack.from_sqlite(
        stack_id="test-lint-prov",
        db_path=tmp_db,
        components=[],
        enforce_provenance=True,
    )


def _make_belief(stack_id: str, statement: str, **overrides) -> Belief:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "statement": statement,
        "belief_type": "fact",
        "confidence": 0.8,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Belief(**defaults)


def _make_value(stack_id: str, name: str, statement: str, **overrides) -> Value:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "name": name,
        "statement": statement,
        "priority": 50,
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Value(**defaults)


def _make_episode(stack_id: str, **overrides) -> Episode:
    defaults = {
        "id": str(uuid.uuid4()),
        "stack_id": stack_id,
        "objective": "Test objective",
        "outcome": "Test outcome",
        "created_at": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return Episode(**defaults)


# ============================================================================
# lint_text — Core Lint Function
# ============================================================================


class TestLintText:
    """Tests for the core lint_text function."""

    def test_valid_text_passes(self):
        result = lint_text("This is a perfectly valid belief statement about the world.")
        assert result.passed is True
        assert result.failures == []

    def test_empty_string_fails_min_length(self):
        result = lint_text("")
        assert result.passed is False
        assert any("too_short" in f for f in result.failures)

    def test_short_string_fails_min_length(self):
        result = lint_text("short")
        assert result.passed is False
        assert any("too_short" in f for f in result.failures)

    def test_exactly_min_length_passes(self):
        result = lint_text("abcdefghij")  # 10 distinct chars
        assert result.passed is True

    def test_one_below_min_length_fails(self):
        result = lint_text("abcdefghi")  # 9 chars
        assert result.passed is False

    def test_max_length_exceeded(self):
        result = lint_text("x" * 2001)
        assert result.passed is False
        assert any("too_long" in f for f in result.failures)

    def test_exactly_max_length_passes(self):
        # Use a repeated phrase (not a single repeated character)
        phrase = "This is a sentence. "
        text = (phrase * (2000 // len(phrase) + 1))[:2000]
        result = lint_text(text)
        assert result.passed is True

    # ---- Prefix artifact detection ----

    def test_prefix_artifact_value_colon(self):
        result = lint_text("Value: something about code quality")
        assert result.passed is False
        assert any("prefix_artifact" in f for f in result.failures)

    def test_prefix_artifact_belief_colon(self):
        result = lint_text("Belief: the system should be reliable")
        assert result.passed is False
        assert any("prefix_artifact" in f for f in result.failures)

    def test_prefix_artifact_statement_colon(self):
        result = lint_text("Statement: I prefer Python over Java")
        assert result.passed is False
        assert any("prefix_artifact" in f for f in result.failures)

    def test_prefix_artifact_title_colon(self):
        result = lint_text("Title: A valid goal that should not have a prefix")
        assert result.passed is False
        assert any("prefix_artifact" in f for f in result.failures)

    def test_prefix_artifact_case_insensitive(self):
        result = lint_text("BELIEF: the system should be reliable")
        assert result.passed is False
        assert any("prefix_artifact" in f for f in result.failures)

    def test_no_prefix_artifact_for_normal_text(self):
        """Text that starts with 'value' but not as a label should pass."""
        result = lint_text("Valuing user feedback leads to better products")
        assert not any("prefix_artifact" in f for f in result.failures)

    # ---- Templated noise detection ----

    def test_templated_noise_worked_on_ellipsis(self):
        result = lint_text("Worked on ...")
        assert result.passed is False
        assert any("templated_noise" in f for f in result.failures)

    def test_templated_noise_updated_ellipsis(self):
        result = lint_text("Updated ...")
        assert result.passed is False
        assert any("templated_noise" in f for f in result.failures)

    def test_templated_noise_just_ellipsis(self):
        result = lint_text("...")
        assert result.passed is False

    def test_templated_noise_unicode_ellipsis(self):
        result = lint_text("\u2026")
        assert result.passed is False

    def test_templated_noise_todo(self):
        result = lint_text("TODO")
        assert result.passed is False
        assert any("templated_noise" in f for f in result.failures)

    def test_templated_noise_placeholder(self):
        result = lint_text("placeholder")
        assert result.passed is False

    def test_templated_noise_repeated_char(self):
        result = lint_text("aaaaa")
        assert result.passed is False

    def test_templated_noise_only_punctuation(self):
        result = lint_text("---!!! ???")
        assert result.passed is False

    def test_templated_noise_n_a(self):
        result = lint_text("n/a")
        assert result.passed is False

    # ---- Fragment detection ----

    def test_trailing_fragment_the(self):
        result = lint_text("I should always consider the")
        assert result.passed is False
        assert any("trailing_fragment" in f for f in result.failures)

    def test_trailing_fragment_and(self):
        result = lint_text("Code quality is important and")
        assert result.passed is False
        assert any("trailing_fragment" in f for f in result.failures)

    def test_trailing_fragment_with(self):
        result = lint_text("Always test code with")
        assert result.passed is False
        assert any("trailing_fragment" in f for f in result.failures)

    def test_trailing_fragment_to(self):
        result = lint_text("The best approach is to")
        assert result.passed is False
        assert any("trailing_fragment" in f for f in result.failures)

    def test_trailing_fragment_of(self):
        result = lint_text("This is a matter of")
        assert result.passed is False
        assert any("trailing_fragment" in f for f in result.failures)

    def test_no_fragment_for_complete_sentence(self):
        result = lint_text("Good code is readable and maintainable.")
        assert result.passed is True

    def test_no_fragment_for_sentence_ending_with_word_containing_article(self):
        """'together' ends with 'the' but is a complete word, not a fragment."""
        result = lint_text("Teams work better together")
        assert result.passed is True  # "together" != " the"

    # ---- Multiple failures ----

    def test_multiple_failures_reported(self):
        result = lint_text("Value: x")
        assert result.passed is False
        assert len(result.failures) >= 2  # too_short + prefix_artifact

    # ---- Configuration ----

    def test_custom_min_length(self):
        config = {**DEFAULT_LINT_CONFIG, "min_length": 5}
        result = lint_text("hello", config)
        assert result.passed is True

    def test_disabled_lint_passes_everything(self):
        config = {**DEFAULT_LINT_CONFIG, "enabled": False}
        result = lint_text("", config)
        assert result.passed is True

    def test_disabled_fragments_check(self):
        config = {**DEFAULT_LINT_CONFIG, "check_fragments": False}
        result = lint_text("This is incomplete and", config)
        assert not any("trailing_fragment" in f for f in result.failures)

    def test_disabled_prefix_artifacts_check(self):
        config = {**DEFAULT_LINT_CONFIG, "check_prefix_artifacts": False}
        result = lint_text("Value: something valid enough in length", config)
        assert not any("prefix_artifact" in f for f in result.failures)

    def test_disabled_templated_noise_check(self):
        config = {**DEFAULT_LINT_CONFIG, "check_templated_noise": False, "min_length": 1}
        result = lint_text("TODO", config)
        assert not any("templated_noise" in f for f in result.failures)


# ============================================================================
# lint_belief
# ============================================================================


class TestLintBelief:
    """Tests for belief-specific linting."""

    def test_valid_belief_passes(self):
        result = lint_belief("Testing code early catches bugs before they compound.")
        assert result.passed is True

    def test_short_belief_fails(self):
        result = lint_belief("short")
        assert result.passed is False

    def test_belief_with_prefix_artifact(self):
        result = lint_belief("Belief: testing is important for code quality")
        assert result.passed is False

    def test_belief_fragment(self):
        result = lint_belief("The most important thing about code quality is")
        assert result.passed is False


# ============================================================================
# lint_value
# ============================================================================


class TestLintValue:
    """Tests for value-specific linting (name + statement)."""

    def test_valid_value_passes(self):
        result = lint_value("Code Quality", "Writing clean, maintainable code is essential.")
        assert result.passed is True

    def test_short_name_fails(self):
        result = lint_value("CQ", "Writing clean, maintainable code is essential.")
        assert result.passed is False
        assert any("name:" in f for f in result.failures)

    def test_short_statement_fails(self):
        result = lint_value("Code Quality", "short")
        assert result.passed is False
        assert any("statement:" in f for f in result.failures)

    def test_both_name_and_statement_fail(self):
        result = lint_value("CQ", "bad")
        assert result.passed is False
        assert any("name:" in f for f in result.failures)
        assert any("statement:" in f for f in result.failures)

    def test_name_prefix_artifact(self):
        result = lint_value(
            "Value: Code Quality", "Writing clean code is essential for long-term success."
        )
        assert result.passed is False
        assert any("name:" in f and "prefix_artifact" in f for f in result.failures)

    def test_statement_fragment(self):
        result = lint_value("Code Quality", "The most important aspect of coding is the")
        assert result.passed is False
        assert any("statement:" in f and "trailing_fragment" in f for f in result.failures)


# ============================================================================
# LintResult
# ============================================================================


class TestLintResult:
    """Tests for LintResult properties."""

    def test_passed_result_properties(self):
        result = LintResult(passed=True)
        assert result.rule_name is None
        assert result.summary == "lint_passed"

    def test_failed_result_properties(self):
        result = LintResult(passed=False, failures=["too_short: content is 3 chars, minimum is 10"])
        assert result.rule_name == "too_short"
        assert "too_short" in result.summary

    def test_multiple_failures_summary(self):
        result = LintResult(
            passed=False,
            failures=[
                "too_short: content is 3 chars",
                "prefix_artifact: starts with label",
            ],
        )
        assert "too_short" in result.summary
        assert "prefix_artifact" in result.summary


# ============================================================================
# get_lint_config
# ============================================================================


class TestGetLintConfig:
    """Tests for configuration loading."""

    def test_default_config_without_getter(self):
        config = get_lint_config()
        assert config == DEFAULT_LINT_CONFIG

    def test_config_from_settings(self):
        def getter(key):
            if key == "lint_rules":
                return json.dumps({"min_length": 20})
            return None

        config = get_lint_config(getter)
        assert config["min_length"] == 20
        assert config["enabled"] is True  # preserved from defaults

    def test_invalid_json_falls_back_to_defaults(self):
        def getter(key):
            if key == "lint_rules":
                return "not valid json"
            return None

        config = get_lint_config(getter)
        assert config == DEFAULT_LINT_CONFIG

    def test_none_setting_uses_defaults(self):
        def getter(key):
            return None

        config = get_lint_config(getter)
        assert config == DEFAULT_LINT_CONFIG


# ============================================================================
# Integration: save_belief with lint
# ============================================================================


class TestSaveBeliefLint:
    """Tests for lint integration in Stack.save_belief."""

    def test_valid_belief_saves_normally(self, stack):
        belief = _make_belief(
            stack.stack_id,
            "Testing code early catches bugs before they compound in complexity.",
        )
        result_id = stack.save_belief(belief)
        assert not result_id.startswith("suggestion:")
        # Verify it was actually saved as a belief
        beliefs = stack.get_beliefs()
        assert len(beliefs) == 1
        assert beliefs[0].statement == belief.statement

    def test_short_belief_redirected_to_suggestion(self, stack):
        belief = _make_belief(stack.stack_id, "short")
        result_id = stack.save_belief(belief)
        assert result_id.startswith("suggestion:")
        # Verify no belief was saved
        beliefs = stack.get_beliefs()
        assert len(beliefs) == 0

    def test_prefix_artifact_belief_redirected(self, stack):
        belief = _make_belief(
            stack.stack_id,
            "Belief: testing is important for code quality and reliability",
        )
        result_id = stack.save_belief(belief)
        assert result_id.startswith("suggestion:")

    def test_fragment_belief_redirected(self, stack):
        belief = _make_belief(
            stack.stack_id,
            "The most important thing about code quality is the",
        )
        result_id = stack.save_belief(belief)
        assert result_id.startswith("suggestion:")

    def test_templated_noise_belief_redirected(self, stack):
        belief = _make_belief(stack.stack_id, "TODO")
        result_id = stack.save_belief(belief)
        assert result_id.startswith("suggestion:")

    def test_redirected_suggestion_has_correct_fields(self, stack):
        belief = _make_belief(stack.stack_id, "bad")
        result_id = stack.save_belief(belief)
        suggestion_id = result_id.replace("suggestion:", "")
        # Get the suggestion from the backend
        suggestion = stack._backend.get_suggestion(suggestion_id)
        assert suggestion is not None
        assert suggestion.memory_type == "belief"
        assert suggestion.status == "rejected"
        assert "lint_failed" in suggestion.resolution_reason
        assert suggestion.content["statement"] == "bad"
        assert suggestion.confidence == 0.0

    def test_lint_failure_logged_to_audit(self, stack):
        belief = _make_belief(stack.stack_id, "bad")
        stack.save_belief(belief)
        audit = stack.get_audit_log(operation="suggestion.resolved")
        # Filter to lint_rejected entries
        lint_entries = []
        for entry in audit:
            d = entry.get("details", {})
            if isinstance(d, str):
                d = json.loads(d)
            if d.get("resolution") == "lint_rejected":
                lint_entries.append(entry)
        assert len(lint_entries) == 1
        entry = lint_entries[0]
        assert entry["operation"] == "suggestion.resolved"
        assert entry["memory_type"] == "belief"
        details = entry.get("details", {})
        if isinstance(details, str):
            details = json.loads(details)
        assert "failures" in details
        assert "redirected_to" in details
        assert details["resolution"] == "lint_rejected"

    def test_lint_disabled_via_settings_saves_normally(self, stack):
        stack.set_stack_setting("lint_rules", json.dumps({"enabled": False}))
        belief = _make_belief(stack.stack_id, "bad")
        result_id = stack.save_belief(belief)
        assert not result_id.startswith("suggestion:")

    def test_custom_min_length_via_settings(self, stack):
        stack.set_stack_setting("lint_rules", json.dumps({"min_length": 3}))
        belief = _make_belief(stack.stack_id, "ok!")
        result_id = stack.save_belief(belief)
        assert not result_id.startswith("suggestion:")

    def test_derived_from_preserved_in_suggestion(self, stack):
        belief = _make_belief(
            stack.stack_id,
            "bad",
            derived_from=["episode:abc123", "note:def456"],
        )
        result_id = stack.save_belief(belief)
        suggestion_id = result_id.replace("suggestion:", "")
        suggestion = stack._backend.get_suggestion(suggestion_id)
        assert "episode:abc123" in suggestion.source_raw_ids
        assert "note:def456" in suggestion.source_raw_ids


# ============================================================================
# Integration: save_value with lint
# ============================================================================


class TestSaveValueLint:
    """Tests for lint integration in Stack.save_value."""

    def test_valid_value_saves_normally(self, stack):
        value = _make_value(
            stack.stack_id,
            "Code Quality",
            "Writing clean, maintainable code is essential for long-term success.",
        )
        result_id = stack.save_value(value)
        assert not result_id.startswith("suggestion:")
        values = stack.get_values()
        assert len(values) == 1

    def test_short_value_name_redirected(self, stack):
        value = _make_value(
            stack.stack_id,
            "CQ",
            "Writing clean, maintainable code is essential for long-term success.",
        )
        result_id = stack.save_value(value)
        assert result_id.startswith("suggestion:")
        values = stack.get_values()
        assert len(values) == 0

    def test_short_value_statement_redirected(self, stack):
        value = _make_value(stack.stack_id, "Code Quality", "short")
        result_id = stack.save_value(value)
        assert result_id.startswith("suggestion:")

    def test_value_name_prefix_artifact_redirected(self, stack):
        value = _make_value(
            stack.stack_id,
            "Value: Code Quality",
            "Writing clean, maintainable code is essential for long-term success.",
        )
        result_id = stack.save_value(value)
        assert result_id.startswith("suggestion:")

    def test_value_statement_fragment_redirected(self, stack):
        value = _make_value(
            stack.stack_id,
            "Code Quality",
            "The most important aspect of coding is the",
        )
        result_id = stack.save_value(value)
        assert result_id.startswith("suggestion:")

    def test_redirected_value_suggestion_has_name_and_statement(self, stack):
        value = _make_value(stack.stack_id, "CQ", "bad")
        result_id = stack.save_value(value)
        suggestion_id = result_id.replace("suggestion:", "")
        suggestion = stack._backend.get_suggestion(suggestion_id)
        assert suggestion is not None
        assert suggestion.memory_type == "value"
        assert suggestion.content["name"] == "CQ"
        assert suggestion.content["statement"] == "bad"

    def test_value_lint_failure_logged_to_audit(self, stack):
        value = _make_value(stack.stack_id, "CQ", "bad")
        stack.save_value(value)
        audit = stack.get_audit_log(operation="suggestion.resolved")
        lint_entries = [
            e
            for e in audit
            if (
                json.loads(e["details"])
                if isinstance(e.get("details"), str)
                else e.get("details", {})
            ).get("resolution")
            == "lint_rejected"
        ]
        assert len(lint_entries) == 1
        assert lint_entries[0]["memory_type"] == "value"


# ============================================================================
# Known-Good Patterns (should always pass)
# ============================================================================


class TestKnownGoodPatterns:
    """Patterns that must always pass lint."""

    @pytest.mark.parametrize(
        "statement",
        [
            "Testing code early catches bugs before they compound in complexity.",
            "Clear variable naming improves readability more than comments do.",
            "Pair programming produces fewer defects per line of code written.",
            "Incremental refactoring is safer than big-bang rewrites of existing systems.",
            "Automated tests provide confidence when making changes to unfamiliar code.",
            "Code reviews distribute knowledge across the team and catch blind spots.",
            "Simple solutions are easier to debug, maintain, and extend over time.",
            "Monitoring production systems reveals problems before users report them.",
            "Documentation decays unless it is treated as code and tested regularly.",
            "Frequent small deployments reduce risk compared to large batched releases.",
        ],
    )
    def test_good_belief_passes(self, statement):
        result = lint_belief(statement)
        assert result.passed is True, f"Good belief rejected: {result.failures}"

    @pytest.mark.parametrize(
        "name,statement",
        [
            (
                "Code Quality",
                "Writing clean, maintainable code is essential for long-term success.",
            ),
            (
                "Collaboration",
                "Working together produces better outcomes than working in isolation.",
            ),
            ("Continuous Learning", "Investing in learning new skills keeps capabilities current."),
            ("User Empathy", "Understanding user needs leads to better product decisions."),
            ("Reliability", "Systems should be designed to handle failures gracefully."),
        ],
    )
    def test_good_value_passes(self, name, statement):
        result = lint_value(name, statement)
        assert result.passed is True, f"Good value rejected: {result.failures}"


# ============================================================================
# Known-Bad Patterns (should always fail)
# ============================================================================


class TestKnownBadPatterns:
    """Patterns that must always fail lint."""

    @pytest.mark.parametrize(
        "statement,expected_failure",
        [
            # Truncated / too short
            ("", "too_short"),
            ("hi", "too_short"),
            ("test", "too_short"),
            # Prefix artifacts
            ("Value: Worked on the API integration pipeline", "prefix_artifact"),
            ("Belief: Code should be clean and readable always", "prefix_artifact"),
            ("Statement: I prefer declarative programming styles", "prefix_artifact"),
            ("Goal: improve the testing infrastructure for CI", "prefix_artifact"),
            ("Name: refactoring the authentication module soon", "prefix_artifact"),
            # Templated noise
            ("Worked on ...", "templated_noise"),
            ("Updated ...", "templated_noise"),
            ("Fixed ...", "templated_noise"),
            ("TODO", "templated_noise"),
            ("placeholder", "templated_noise"),
            ("n/a", "templated_noise"),
            # Trailing fragments
            ("The system should always handle the", "trailing_fragment"),
            ("We need to focus on improving the quality of", "trailing_fragment"),
            ("This feature was designed to work with", "trailing_fragment"),
            ("The primary benefit of this approach is", "trailing_fragment"),
        ],
    )
    def test_bad_belief_fails(self, statement, expected_failure):
        result = lint_belief(statement)
        assert result.passed is False, f"Bad belief passed: '{statement}'"
        assert any(
            expected_failure in f for f in result.failures
        ), f"Expected '{expected_failure}' failure, got: {result.failures}"

    @pytest.mark.parametrize(
        "name,statement,expected_failure",
        [
            ("CQ", "Writing clean code is essential for long-term success.", "name:"),
            ("x", "A valid statement about code quality and maintainability.", "name:"),
            ("Value: Quality", "Valid statement about software engineering practices.", "name:"),
            ("Quality", "bad", "statement:"),
            ("Quality", "Belief: something about code quality standards", "statement:"),
        ],
    )
    def test_bad_value_fails(self, name, statement, expected_failure):
        result = lint_value(name, statement)
        assert result.passed is False, f"Bad value passed: name='{name}', statement='{statement}'"
        assert any(
            expected_failure in f for f in result.failures
        ), f"Expected '{expected_failure}' failure, got: {result.failures}"


# ============================================================================
# Lifecycle state interaction
# ============================================================================


class TestLintLifecycleState:
    """Test that lint only runs in ACTIVE state, not INITIALIZING."""

    def test_lint_skipped_in_initializing_state(self, tmp_path):
        """In INITIALIZING state (seed data), lint should NOT run."""
        db = tmp_path / "test_init.db"
        stack = Stack.from_sqlite(
            stack_id="test-init",
            db_path=db,
            components=[],
            enforce_provenance=False,
        )
        # Stack starts in INITIALIZING — lint should be skipped
        belief = _make_belief(stack.stack_id, "bad")
        result_id = stack.save_belief(belief)
        # Should save normally, not redirect
        assert not result_id.startswith("suggestion:")

    def test_lint_runs_after_attach(self, tmp_path):
        """After on_attach, stack transitions to ACTIVE and lint runs."""
        db = tmp_path / "test_attach.db"
        stack = Stack.from_sqlite(
            stack_id="test-attach",
            db_path=db,
            components=[],
            enforce_provenance=False,
        )
        stack.on_attach("test-core")
        belief = _make_belief(stack.stack_id, "bad")
        result_id = stack.save_belief(belief)
        assert result_id.startswith("suggestion:")


# ============================================================================
# Provenance + Lint interaction
# ============================================================================


class TestProvenanceLintInteraction:
    """Test that provenance validation runs before lint (provenance errors
    should take priority over lint failures)."""

    def test_provenance_error_before_lint(self, stack_with_provenance):
        """If provenance is enforced, provenance errors fire before lint."""
        stack = stack_with_provenance
        # Transition from INITIALIZING to ACTIVE
        stack.on_attach("test-core")
        belief = _make_belief(stack.stack_id, "bad", derived_from=None)
        # Should get provenance error, not lint redirect
        from kernle.protocols import ProvenanceError

        with pytest.raises(ProvenanceError):
            stack.save_belief(belief)
