"""Additional tests for kernle/cli/__main__.py to improve coverage.

Targets uncovered lines: validate_input errors, cmd_init, cmd_mcp,
raw argv preprocessing, plugin discovery, dispatch branches, error handling.
"""

import argparse
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kernle.cli.__main__ import cmd_init, main, validate_input
from kernle.core import Kernle
from tests.conftest import bind_noop_model

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def storage(tmp_path, sqlite_storage_factory):
    return sqlite_storage_factory(stack_id="test-main", db_path=tmp_path / "main.db")


@pytest.fixture
def k(storage):
    inst = Kernle(stack_id="test-main", storage=storage, strict=False)
    bind_noop_model(inst)
    yield inst


# ============================================================================
# validate_input
# ============================================================================


class TestValidateInput:
    """Tests for the validate_input function."""

    def test_non_string_raises(self):
        """Non-string input raises ValueError."""
        with pytest.raises(ValueError, match="must be a string"):
            validate_input(123, "test_field")

    def test_too_long_raises(self):
        """Input exceeding max_length raises ValueError."""
        with pytest.raises(ValueError, match="too long"):
            validate_input("x" * 1001, "test_field")

    def test_too_long_custom_max(self):
        """Custom max_length is respected."""
        with pytest.raises(ValueError, match="too long"):
            validate_input("x" * 51, "test_field", max_length=50)

    def test_valid_input_passes(self):
        """Normal input passes validation and is returned."""
        result = validate_input("hello world", "test_field")
        assert result == "hello world"

    def test_control_characters_stripped(self):
        """Control characters are removed from input."""
        result = validate_input("hello\x00world\x07!", "test_field")
        assert result == "helloworld!"

    def test_newlines_preserved(self):
        """Newlines are NOT stripped (only control chars except newlines)."""
        result = validate_input("hello\nworld", "test_field")
        assert result == "hello\nworld"


# ============================================================================
# cmd_init (lines 93-423)
# ============================================================================


class TestCmdInit:
    """Tests for the cmd_init function — full setup wizard."""

    def test_init_non_interactive_claude_code(self, k, capsys):
        """Non-interactive init with claude-code env produces correct output."""
        args = argparse.Namespace(
            non_interactive=True,
            env="claude-code",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Welcome to Kernle" in captured
        assert "Claude Code Setup" in captured
        assert "MCP server" in captured
        assert "Setup Complete" in captured

    def test_init_non_interactive_openclaw(self, k, capsys):
        """Non-interactive init with openclaw env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="openclaw",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "OpenClaw Setup" in captured
        assert "AGENTS.md" in captured

    def test_init_non_interactive_cline(self, k, capsys):
        """Non-interactive init with cline env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="cline",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Cline Setup" in captured
        assert ".clinerules" in captured

    def test_init_non_interactive_cursor(self, k, capsys):
        """Non-interactive init with cursor env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="cursor",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Cursor Setup" in captured
        assert ".cursorrules" in captured

    def test_init_non_interactive_desktop(self, k, capsys):
        """Non-interactive init with desktop env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="desktop",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Claude Desktop Setup" in captured

    def test_init_non_interactive_other(self, k, capsys):
        """Non-interactive init with other/manual env."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Manual Setup" in captured
        assert "CLI commands" in captured

    def test_init_with_seed_values(self, k, capsys):
        """Init with --seed-values creates default values."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=True,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Seeding Initial Values" in captured
        assert len(k._storage.get_values()) >= 1

    def test_init_seed_values_skip_existing(self, k, capsys):
        """Init skips seeding when values already exist."""
        # Create an existing value first
        k.value("existing_value", "Already exists", priority=50)

        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=True,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "existing values, skipping seed" in captured

    def test_init_interactive_env_selection(self, k, capsys):
        """Interactive env selection uses input()."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        # Simulate user choosing "6" (other)
        with patch("builtins.input", return_value="6"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Manual Setup" in captured

    def test_init_interactive_env_eof(self, k, capsys):
        """Interactive env selection handles EOFError gracefully."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        with patch("builtins.input", side_effect=EOFError):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Aborted" in captured

    def test_init_interactive_id_selection(self, k, capsys):
        """Interactive ID selection with auto-generated stack_id."""
        # Give the kernle instance an auto-generated stack_id
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value="my-agent"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Using: my-agent" in captured

    def test_init_interactive_id_invalid(self, k, capsys):
        """Interactive ID selection rejects invalid characters."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value="invalid name!"):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Invalid" in captured

    def test_init_interactive_id_empty(self, k, capsys):
        """Interactive ID selection keeps auto when empty input."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", return_value=""):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "auto-abc123" in captured

    def test_init_interactive_id_eof(self, k, capsys):
        """Interactive ID selection handles EOFError."""
        k.stack_id = "auto-abc123"

        args = argparse.Namespace(
            non_interactive=False,
            env="other",
            seed_values=False,
        )
        with patch("builtins.input", side_effect=EOFError):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        # Should still complete
        assert "Setup Complete" in captured

    def test_init_checkpoint_error(self, k, capsys):
        """Init handles checkpoint creation failure gracefully."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        with patch.object(k, "checkpoint", side_effect=Exception("DB error")):
            cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Warning" in captured
        assert "Setup Complete" in captured

    def test_init_seed_trust(self, k, capsys):
        """Init calls seed_trust and reports count."""
        args = argparse.Namespace(
            non_interactive=True,
            env="other",
            seed_values=False,
        )
        cmd_init(args, k)
        captured = capsys.readouterr().out
        # seed_trust returns count of seeded assessments
        assert "Setup Complete" in captured
        assert "trust assessments" in captured

    def test_init_env_detection(self, k, capsys, tmp_path):
        """Init detects environment files when none is specified."""
        args = argparse.Namespace(
            non_interactive=False,
            env=None,
            seed_values=False,
        )
        # Create CLAUDE.md to trigger detection
        with patch("builtins.input", return_value="1"):
            with patch.object(Path, "exists", return_value=True):
                cmd_init(args, k)
        captured = capsys.readouterr().out
        assert "Claude Code Setup" in captured


# ============================================================================
# cmd_mcp (lines 426-434)
# ============================================================================


class TestCmdInitPostConditions:
    """Focused tests for commands.init cmd_init post-condition checks."""

    def _base_args(self, **overrides):
        defaults = {
            "style": "standard",
            "output": None,
            "print": False,
            "force": False,
            "no_per_message": False,
            "non_interactive": True,
            "seed_values": False,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_cmd_init_returns_structured_success(self, k, tmp_path, monkeypatch):
        """Init returns structured success information after file write."""
        from kernle.cli.commands.init import cmd_init as cmd_init_file

        monkeypatch.chdir(tmp_path)
        args = self._base_args()

        result = cmd_init_file(args, k)

        assert isinstance(result, dict)
        assert result["success"] is True
        assert result["status"] == "success"
        assert result["action"] == "create"
        assert result["checks"]["file"]["exists"] is True
        assert result["checks"]["file"]["post_has_section"] is True
        assert result["checks"]["state"]["checkpoint"]["saved"] is True

    def test_cmd_init_state_seed_values_checked(self, k, tmp_path, monkeypatch):
        """Seed-value request enforces value-side post-conditions."""
        from kernle.cli.commands.init import cmd_init as cmd_init_file

        monkeypatch.chdir(tmp_path)
        args = self._base_args(seed_values=True)

        result = cmd_init_file(args, k)

        assert result["success"] is True
        state = result["checks"]["state"]
        assert state["seed_values"]["requested"] is True
        assert state["seed_values"]["required_present"] is True

    def test_cmd_init_existing_instructions_returns_failure(self, k, tmp_path, monkeypatch):
        """cmd_init returns structured non-success when instructions already exist."""
        from kernle.cli.commands.init import cmd_init as cmd_init_file

        monkeypatch.chdir(tmp_path)
        (tmp_path / "CLAUDE.md").write_text("# Memory\n\n## Memory (Kernle)\n\nAlready configured.")

        args = self._base_args()
        result = cmd_init_file(args, k)

        assert result["success"] is False
        assert result["status"] == "already_present"
        assert result["action"] == "skip"


class TestCmdMcp:
    """Tests for the cmd_mcp function."""

    def test_mcp_dispatches_to_server(self):
        """cmd_mcp calls the MCP server main function."""
        args = argparse.Namespace(stack="my-agent")

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                cmd_mcp(args)
                mock_mcp_main.assert_called_once_with(stack_id="my-agent")

    def test_mcp_default_stack(self):
        """cmd_mcp uses 'default' when no stack specified."""
        args = argparse.Namespace()  # No stack attribute

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                cmd_mcp(args)
                mock_mcp_main.assert_called_once_with(stack_id="default")

    def test_mcp_empty_stack_is_rejected(self):
        """cmd_mcp rejects empty/whitespace stack IDs."""
        args = argparse.Namespace(stack="   ")

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                with pytest.raises(SystemExit) as exc:
                    cmd_mcp(args)
                assert exc.value.code == 2
                mock_mcp_main.assert_not_called()

    def test_mcp_invalid_stack_sanitization_error(self):
        """cmd_mcp sanitizes control characters in stack IDs."""
        args = argparse.Namespace(stack="bad\0id")

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                cmd_mcp(args)
                mock_mcp_main.assert_called_once_with(stack_id="badid")

    def test_mcp_non_string_stack_is_rejected(self):
        """cmd_mcp rejects non-string stack values."""
        args = argparse.Namespace(stack=123)

        with patch("kernle.cli.__main__.sys") as mock_sys:
            mock_sys.stderr = StringIO()
            with patch("kernle.mcp.server.main") as mock_mcp_main:
                from kernle.cli.__main__ import cmd_mcp

                with pytest.raises(SystemExit) as exc:
                    cmd_mcp(args)
                assert exc.value.code == 2
                mock_mcp_main.assert_not_called()


# ============================================================================
# Raw argv preprocessing (lines 1954-1987)
# ============================================================================


class TestRawArgvPreprocessing:
    """Tests for the raw subcommand argv preprocessing."""

    def test_raw_content_preprocessing_runs(self, k, capsys):
        """'kernle raw "content"' triggers the argv preprocessing (inserts capture).

        Note: The inserted 'capture' is consumed by argparse as the content
        positional, so the parse actually fails. But the preprocessing code
        path (lines 1979-1986) IS exercised.
        """
        test_args = ["kernle", "raw", "some content"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with pytest.raises(SystemExit):
                        main()

    def test_raw_list_no_insert(self, k, capsys):
        """'kernle raw list' does NOT insert 'capture'."""
        test_args = ["kernle", "raw", "list"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

        # Should run raw list without error (output may be empty list)
        captured = capsys.readouterr().out
        assert "No raw entries found." in captured

    def test_raw_with_stack_flag_before(self, k, capsys):
        """'kernle --stack X raw "content"' preprocessing correctly skips flag."""
        test_args = ["kernle", "--stack", "test-main", "raw", "test content"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                # The preprocessing still triggers the same argparse issue
                with pytest.raises(SystemExit):
                    main()

    def test_raw_with_dash_arg(self, k, capsys):
        """'kernle raw --quiet' does NOT insert 'capture' (starts with -)."""
        test_args = ["kernle", "raw", "--quiet"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    # No content arg but quiet mode, so cmd_raw handles it
                    main()

    def test_raw_no_content(self, k, capsys):
        """'kernle raw' with no content or subcommand still dispatches."""
        test_args = ["kernle", "raw"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

        # Should dispatch to cmd_raw with raw_action=None and content=None
        captured = capsys.readouterr().out
        assert "Content is required" in captured


# ============================================================================
# Plugin discovery (lines 1990-2005)
# ============================================================================


class TestPluginDiscovery:
    """Tests for plugin CLI discovery in main()."""

    def test_plugin_discovery_failure_logged(self, k, capsys):
        """Plugin discovery failure is logged but doesn't crash main()."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.discovery.discover_plugins",
                        side_effect=Exception("Discovery failed"),
                    ):
                        main()

        # Should still complete the status command
        captured = capsys.readouterr().out
        assert "Memory Status" in captured

    def test_plugin_cli_registration_failure(self, k, capsys):
        """Individual plugin registration failure doesn't crash."""
        test_args = ["kernle", "status"]

        mock_comp = MagicMock()
        mock_comp.name = "broken-plugin"

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch("kernle.discovery.discover_plugins", return_value=[mock_comp]):
                        with patch(
                            "kernle.discovery.load_component",
                            side_effect=ImportError("bad plugin"),
                        ):
                            main()

        captured = capsys.readouterr().out
        assert "Memory Status" in captured


# ============================================================================
# Dispatch branches (lines 2027-2138)
# ============================================================================


class TestDispatchBranches:
    """Tests for command dispatch in main() — covering uncovered branches."""

    def _run_main(self, argv, k):
        """Helper to run main() with given argv and kernle instance."""
        with patch("sys.argv", ["kernle"] + argv):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    main()

    @pytest.mark.parametrize(
        ("argv", "patch_target"),
        [
            (["extract", "summary of conversation"], "kernle.cli.__main__.cmd_extract"),
            (["resume"], "kernle.cli.__main__.cmd_resume"),
            (["init", "--non-interactive", "-y"], "kernle.cli.__main__.cmd_init_md"),
            (["doctor"], "kernle.cli.__main__.cmd_doctor"),
            (["doctor", "structural"], "kernle.cli.__main__.cmd_doctor_structural"),
            (["relation", "list"], "kernle.cli.__main__.cmd_relation"),
            (["entity-model", "list"], "kernle.cli.__main__.cmd_entity_model"),
            (["emotion", "summary"], "kernle.cli.__main__.cmd_emotion"),
            (["meta", "uncertain"], "kernle.cli.__main__.cmd_meta"),
            (["anxiety"], "kernle.cli.__main__.cmd_anxiety"),
            (["stats", "health-checks"], "kernle.cli.__main__.cmd_stats"),
            (["forget", "candidates"], "kernle.cli.__main__.cmd_forget"),
            (["epoch", "list"], "kernle.cli.__main__.cmd_epoch"),
            (["summary", "list"], "kernle.cli.__main__.cmd_summary"),
            (["narrative", "show"], "kernle.cli.__main__.cmd_narrative"),
            (["playbook", "list"], "kernle.cli.__main__.cmd_playbook"),
            (["process", "status"], "kernle.cli.__main__.cmd_process"),
            (["suggestions", "list"], "kernle.cli.__main__.cmd_suggestions"),
            (["belief", "list"], "kernle.cli.__main__.cmd_belief"),
            (["dump"], "kernle.cli.__main__.cmd_dump"),
            (["export", "export.json", "--format", "json"], "kernle.cli.__main__.cmd_export"),
            (["export-cache"], "kernle.cli.__main__.cmd_export_cache"),
            (["export-full"], "kernle.cli.__main__.cmd_export_full"),
            (["boot", "list"], "kernle.cli.__main__.cmd_boot"),
            (["sync", "conflicts"], "kernle.cli.__main__.cmd_sync"),
            (["auth", "status"], "kernle.cli.__main__.cmd_auth"),
            (["import", "import.md", "--dry-run"], "kernle.cli.__main__.cmd_import"),
            (["migrate", "seed-beliefs", "--dry-run"], "kernle.cli.__main__.cmd_migrate"),
            (["setup", "claude-code"], "kernle.cli.__main__.cmd_setup"),
            (["search", "test query"], "kernle.cli.__main__.cmd_search"),
            (["when", "today"], "kernle.cli.__main__.cmd_temporal"),
            (["promote"], "kernle.cli.__main__.cmd_promote"),
            (["migrate", "backfill-provenance"], "kernle.cli.__main__.cmd_migrate"),
            (["migrate", "backfill-provenance", "--json"], "kernle.cli.__main__.cmd_migrate"),
            (["migrate", "link-raw", "--window", "15"], "kernle.cli.__main__.cmd_migrate"),
        ],
    )
    def test_dispatch_routes_to_handler(self, argv, patch_target, k):
        with patch(patch_target) as mock_handler:
            self._run_main(argv, k)

        mock_handler.assert_called_once()
        call_args = mock_handler.call_args.args
        assert len(call_args) == 2
        assert call_args[1] is k

    def test_dispatch_doctor_session_start_gate(self, k, capsys):
        """Dispatch 'doctor session start' shows devtools install message when not installed."""
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=SystemExit(2),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(["doctor", "session", "start"], k)
        assert exc.value.code == 2

    def test_dispatch_doctor_session_list_gate(self, k, capsys):
        """Dispatch 'doctor session list' shows devtools install message when not installed."""
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=SystemExit(2),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(["doctor", "session", "list"], k)
        assert exc.value.code == 2

    def test_dispatch_doctor_session_start_import_error_propagates(self, k):
        """Dispatch 'doctor session start' propagates bare ImportError (not swallowed).

        Bare ImportError is NOT caught by the devtools gate (exit 2). Instead it
        propagates to main()'s general exception handler (exit 1).
        """
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=ImportError("kernle-devtools requires kernle>=0.12.4"),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(["doctor", "session", "start"], k)
            # Exit code 1 (general error), NOT 2 (missing devtools)
            assert exc.value.code == 1

    def test_dispatch_doctor_session_no_action(self, k, capsys):
        """Dispatch 'doctor session' without start/list shows usage."""
        self._run_main(["doctor", "session"], k)
        captured = capsys.readouterr().out
        assert "Usage" in captured

    def test_dispatch_doctor_report_gate(self, k, capsys):
        """Dispatch 'doctor report' shows devtools install message when not installed."""
        with patch(
            "kernle.cli.__main__._import_devtools",
            side_effect=SystemExit(2),
        ):
            with pytest.raises(SystemExit) as exc:
                self._run_main(["doctor", "report", "latest"], k)
        assert exc.value.code == 2

    def test_dispatch_trust(self, k):
        """Dispatch 'trust' command."""
        with patch("kernle.cli.commands.trust.cmd_trust") as mock_trust:
            self._run_main(["trust", "list"], k)

        mock_trust.assert_called_once()
        call_args = mock_trust.call_args.args
        assert len(call_args) == 2
        assert call_args[1] is k

    def test_dispatch_identity_default(self, k):
        """Dispatch 'identity' with no subcommand defaults to show."""
        with patch("kernle.cli.__main__.cmd_identity") as mock_identity:
            self._run_main(["identity"], k)

        mock_identity.assert_called_once()
        parsed_args = mock_identity.call_args.args[0]
        assert parsed_args.identity_action == "show"
        assert mock_identity.call_args.args[1] is k

    def test_dispatch_identity_show(self, k):
        """Dispatch 'identity show'."""
        with patch("kernle.cli.__main__.cmd_identity") as mock_identity:
            self._run_main(["identity", "show"], k)

        mock_identity.assert_called_once()
        parsed_args = mock_identity.call_args.args[0]
        assert parsed_args.identity_action == "show"
        assert mock_identity.call_args.args[1] is k

    def test_dispatch_stack_is_hermetic(self, k, monkeypatch, tmp_path):
        """Dispatch 'stack' command without touching ambient ~/.kernle state."""
        isolated_data_dir = tmp_path / "kernle-data"
        isolated_data_dir.mkdir()
        monkeypatch.setenv("KERNLE_DATA_DIR", str(isolated_data_dir))

        with patch("kernle.cli.__main__.cmd_stack") as mock_stack:
            self._run_main(["stack", "list"], k)

        mock_stack.assert_called_once()
        call_args = mock_stack.call_args.args
        assert len(call_args) == 2
        assert call_args[1] is k

    def test_dispatch_mcp(self, k):
        """Dispatch 'mcp' command through main()."""
        with patch("kernle.cli.__main__.cmd_mcp") as mock_mcp:
            self._run_main(["mcp"], k)

        mock_mcp.assert_called_once()


# ============================================================================
# Error handling in main() (lines 2133-2138)
# ============================================================================


class TestMainErrorHandling:
    """Test error handling in main() dispatch."""

    def test_value_error_exits(self, k):
        """ValueError during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=ValueError("bad input"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_type_error_exits(self, k):
        """TypeError during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=TypeError("wrong type"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_generic_exception_exits(self, k):
        """Generic Exception during command execution causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.Kernle", return_value=k):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test-main"):
                    with patch(
                        "kernle.cli.__main__.cmd_status",
                        side_effect=RuntimeError("unexpected"),
                    ):
                        with pytest.raises(SystemExit) as exc_info:
                            main()
                        assert exc_info.value.code == 1

    def test_init_type_error_exits(self):
        """TypeError during Kernle initialization causes sys.exit(1)."""
        test_args = ["kernle", "status"]

        with patch("sys.argv", test_args):
            with patch(
                "kernle.cli.__main__.Kernle",
                side_effect=TypeError("bad init"),
            ):
                with patch("kernle.cli.__main__.resolve_stack_id", return_value="test"):
                    with pytest.raises(SystemExit) as exc_info:
                        main()
                    assert exc_info.value.code == 1


# ============================================================================
# Hook dispatch (lines 2011-2013)
# ============================================================================


class TestHookDispatch:
    """Test hook command dispatch in main()."""

    def test_hook_dispatch_before_kernle_init(self):
        """Hook commands are dispatched BEFORE Kernle initialization."""
        test_args = ["kernle", "hook", "session-start"]

        with patch("sys.argv", test_args):
            with patch("kernle.cli.__main__.cmd_hook") as mock_hook:
                mock_hook.return_value = None
                # Kernle should NOT be initialized for hook commands
                with patch("kernle.cli.__main__.Kernle") as mock_kernle:
                    main()
                    mock_hook.assert_called_once()
                    mock_kernle.assert_not_called()
