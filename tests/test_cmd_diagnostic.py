"""Tests for kernle diagnostic CLI commands — status, resume, temporal, dump,
export, export_full, export_cache, boot, drive."""

import json
from argparse import Namespace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kernle import Kernle
from kernle.cli.commands.diagnostic import (
    cmd_boot,
    cmd_drive,
    cmd_dump,
    cmd_export,
    cmd_export_cache,
    cmd_export_full,
    cmd_resume,
    cmd_status,
    cmd_temporal,
)
from kernle.storage import SQLiteStorage
from tests.conftest import bind_noop_model

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def k(tmp_path):
    storage = SQLiteStorage(stack_id="test-diag", db_path=tmp_path / "diag.db")
    inst = Kernle(
        stack_id="test-diag",
        storage=storage,
        strict=False,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    bind_noop_model(inst)
    yield inst
    storage.close()


def _args(**kwargs):
    return Namespace(**kwargs)


# ===========================================================================
# cmd_status
# ===========================================================================


class TestCmdStatus:
    def test_empty_memory(self, k, capsys):
        cmd_status(_args(), k)
        out = capsys.readouterr().out
        assert "Memory Status for test-diag" in out
        assert "Values:     0" in out
        assert "Beliefs:    0" in out
        assert "Goals:      0 active" in out
        assert "Episodes:   0" in out
        assert "Checkpoint: No" in out

    def test_populated_memory(self, k, capsys):
        k.value("honesty", "I value honesty")
        k.belief("The sky is blue", confidence=0.9)
        k.goal("Learn testing")
        k.episode("Did a thing", "success")
        cmd_status(_args(), k)
        out = capsys.readouterr().out
        assert "Values:     1" in out
        assert "Beliefs:    1" in out
        assert "Goals:      1 active" in out
        assert "Episodes:   1" in out

    def test_checkpoint_shown(self, k, capsys):
        k.checkpoint("working on tests")
        cmd_status(_args(), k)
        out = capsys.readouterr().out
        assert "Checkpoint: Yes" in out

    def test_raw_count_shown(self, k, capsys):
        k.raw("raw entry")
        cmd_status(_args(), k)
        out = capsys.readouterr().out
        assert "Raw:        1" in out


# ===========================================================================
# cmd_resume
# ===========================================================================


class TestCmdResume:
    def test_no_checkpoint(self, k, capsys):
        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "No checkpoint found" in out

    def test_with_checkpoint(self, k, capsys):
        k.checkpoint("fixing auth bug")
        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "Resume Point" in out
        assert "fixing auth bug" in out

    def test_pending_items(self, k, capsys):
        k.checkpoint("task", pending=["item1", "item2", "item3", "item4"])
        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "Pending (4 items)" in out
        assert "item1" in out
        assert "item2" in out
        assert "item3" in out
        assert "... and 1 more" in out

    def test_context_parsed(self, k, capsys):
        k.checkpoint(
            "task",
            context="Progress: 50% done | Next: write tests | Blocker: need API key",
        )
        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "Progress: 50% done" in out
        assert "Next: write tests" in out
        assert "Blocker: need API key" in out

    def test_age_display_just_now(self, k, capsys):
        k.checkpoint("task")
        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "just now" in out

    def test_stale_checkpoint_warning(self, k, capsys):
        k.checkpoint("old task")
        cp_file = k.checkpoint_dir / "test-diag.json"
        data = json.loads(cp_file.read_text())
        old_time = (datetime.now(timezone.utc) - timedelta(hours=8)).isoformat()
        if isinstance(data, list):
            data[-1]["timestamp"] = old_time
        else:
            data["timestamp"] = old_time
        cp_file.write_text(json.dumps(data))

        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "Checkpoint is stale" in out

    def test_age_display_hours(self, k, capsys):
        k.checkpoint("task")
        cp_file = k.checkpoint_dir / "test-diag.json"
        data = json.loads(cp_file.read_text())
        old_time = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        if isinstance(data, list):
            data[-1]["timestamp"] = old_time
        else:
            data["timestamp"] = old_time
        cp_file.write_text(json.dumps(data))

        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "3h ago" in out

    def test_age_display_minutes(self, k, capsys):
        k.checkpoint("task")
        cp_file = k.checkpoint_dir / "test-diag.json"
        data = json.loads(cp_file.read_text())
        old_time = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        if isinstance(data, list):
            data[-1]["timestamp"] = old_time
        else:
            data["timestamp"] = old_time
        cp_file.write_text(json.dumps(data))

        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "15m ago" in out

    def test_age_display_days(self, k, capsys):
        k.checkpoint("task")
        cp_file = k.checkpoint_dir / "test-diag.json"
        data = json.loads(cp_file.read_text())
        old_time = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        if isinstance(data, list):
            data[-1]["timestamp"] = old_time
        else:
            data["timestamp"] = old_time
        cp_file.write_text(json.dumps(data))

        cmd_resume(_args(), k)
        out = capsys.readouterr().out
        assert "2d ago" in out


# ===========================================================================
# cmd_temporal
# ===========================================================================


class TestCmdTemporal:
    def test_today_empty(self, k, capsys):
        cmd_temporal(_args(when="today"), k)
        out = capsys.readouterr().out
        assert "What happened today" in out
        assert "Time range:" in out

    def test_yesterday(self, k, capsys):
        cmd_temporal(_args(when="yesterday"), k)
        out = capsys.readouterr().out
        assert "What happened yesterday" in out

    def test_week(self, k, capsys):
        cmd_temporal(_args(when="this week"), k)
        out = capsys.readouterr().out
        assert "What happened this week" in out

    def test_hour(self, k, capsys):
        cmd_temporal(_args(when="last hour"), k)
        out = capsys.readouterr().out
        assert "What happened last hour" in out

    def test_with_episodes(self, k, capsys):
        k.episode("Deployed v2", "success", lessons=["Worked well"])
        cmd_temporal(_args(when="today"), k)
        out = capsys.readouterr().out
        assert "Episodes:" in out
        assert "Deployed v2" in out

    def test_with_notes(self, k, capsys):
        k.note("Interesting finding about caching")
        cmd_temporal(_args(when="today"), k)
        out = capsys.readouterr().out
        assert "Notes:" in out
        assert "Interesting finding" in out


# ===========================================================================
# cmd_dump
# ===========================================================================


class TestCmdDump:
    def test_markdown_format(self, k, capsys):
        k.value("integrity", "Act with integrity")
        cmd_dump(_args(include_raw=False, format="markdown"), k)
        out = capsys.readouterr().out
        assert "Memory Dump for test-diag" in out
        assert "integrity" in out

    def test_json_format(self, k, capsys):
        k.value("integrity", "Act with integrity")
        cmd_dump(_args(include_raw=False, format="json"), k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["stack_id"] == "test-diag"

    def test_include_raw(self, k, capsys):
        k.raw("some raw data")
        cmd_dump(_args(include_raw=True, format="markdown"), k)
        out = capsys.readouterr().out
        assert "Raw" in out


# ===========================================================================
# cmd_export
# ===========================================================================


class TestCmdExport:
    def test_export_markdown(self, k, capsys, tmp_path):
        k.value("courage", "Be courageous")
        path = str(tmp_path / "export.md")
        cmd_export(_args(path=path, include_raw=False, format="markdown"), k)
        out = capsys.readouterr().out
        assert f"Exported memory to {path}" in out
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "courage" in content

    def test_export_json(self, k, capsys, tmp_path):
        k.belief("Testing works", confidence=0.9)
        path = str(tmp_path / "export.json")
        cmd_export(_args(path=path, include_raw=False, format="json"), k)
        out = capsys.readouterr().out
        assert f"Exported memory to {path}" in out
        content = json.loads(Path(path).read_text())
        assert content["stack_id"] == "test-diag"

    def test_auto_detect_json_from_extension(self, k, capsys, tmp_path):
        """When format is None, auto-detect from .json extension."""
        path = str(tmp_path / "export.json")
        cmd_export(_args(path=path, include_raw=False, format=None), k)
        out = capsys.readouterr().out
        assert f"Exported memory to {path}" in out
        content = json.loads(Path(path).read_text())
        assert "stack_id" in content

    def test_auto_detect_markdown_from_extension(self, k, capsys, tmp_path):
        """When format is None and extension is .md, use markdown."""
        path = str(tmp_path / "export.md")
        cmd_export(_args(path=path, include_raw=False, format=None), k)
        out = capsys.readouterr().out
        assert f"Exported memory to {path}" in out
        content = Path(path).read_text()
        assert "Memory Dump" in content


# ===========================================================================
# cmd_export_full
# ===========================================================================


class TestCmdExportFull:
    def test_to_stdout_markdown(self, k, capsys):
        k.value("curiosity", "Stay curious")
        cmd_export_full(_args(path=None, format="markdown", include_raw=False), k)
        out = capsys.readouterr().out
        assert "curiosity" in out

    def test_to_stdout_json(self, k, capsys):
        k.value("growth", "Always grow")
        cmd_export_full(_args(path=None, format="json", include_raw=False), k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "values" in parsed

    def test_to_file(self, k, capsys, tmp_path):
        k.belief("File exports work", confidence=0.8)
        path = str(tmp_path / "full.md")
        cmd_export_full(_args(path=path, format="markdown", include_raw=False), k)
        out = capsys.readouterr().out
        assert f"Exported full agent context to {path}" in out
        assert Path(path).exists()

    def test_auto_detect_json(self, k, capsys, tmp_path):
        path = str(tmp_path / "full.json")
        cmd_export_full(_args(path=path, format=None, include_raw=False), k)
        out = capsys.readouterr().out
        assert f"Exported full agent context to {path}" in out
        content = json.loads(Path(path).read_text())
        assert "stack_id" in content

    def test_auto_detect_markdown_default(self, k, capsys):
        """When path is None and format is None, defaults to markdown."""
        cmd_export_full(_args(path=None, format=None, include_raw=False), k)
        out = capsys.readouterr().out
        # Markdown output — not JSON
        assert "# " in out


# ===========================================================================
# cmd_export_cache
# ===========================================================================


class TestCmdExportCache:
    def test_to_stdout(self, k, capsys):
        k.belief("Caching is useful", confidence=0.9)
        cmd_export_cache(
            _args(output=None, min_confidence=0.4, max_beliefs=50, no_checkpoint=False),
            k,
        )
        out = capsys.readouterr().out
        assert "MEMORY.md" in out
        assert "Caching is useful" in out

    def test_to_file(self, k, capsys, tmp_path):
        k.value("speed", "Be fast")
        k.belief("Tests are essential", confidence=0.8)
        path = str(tmp_path / "cache.md")
        cmd_export_cache(
            _args(output=path, min_confidence=0.4, max_beliefs=50, no_checkpoint=False),
            k,
        )
        out = capsys.readouterr().out
        assert f"Cache exported to {path}" in out
        assert "Beliefs:" in out
        assert "Min confidence: 40%" in out

    def test_no_checkpoint_flag(self, k, capsys):
        k.checkpoint("some task")
        cmd_export_cache(
            _args(output=None, min_confidence=0.4, max_beliefs=50, no_checkpoint=True),
            k,
        )
        out = capsys.readouterr().out
        assert "MEMORY.md" in out

    def test_min_confidence_filter(self, k, capsys):
        k.belief("Low confidence fact", confidence=0.2)
        k.belief("High confidence fact", confidence=0.9)
        cmd_export_cache(
            _args(output=None, min_confidence=0.5, max_beliefs=50, no_checkpoint=False),
            k,
        )
        out = capsys.readouterr().out
        assert "High confidence fact" in out
        assert "Low confidence fact" not in out


# ===========================================================================
# cmd_boot
# ===========================================================================


class TestCmdBoot:
    def test_set(self, k, capsys):
        cmd_boot(_args(boot_action="set", key="model", value="claude-3"), k)
        out = capsys.readouterr().out
        assert "model: claude-3" in out

    def test_get_existing(self, k, capsys):
        k.boot_set("model", "claude-3")
        cmd_boot(_args(boot_action="get", key="model"), k)
        out = capsys.readouterr().out
        assert "claude-3" in out

    def test_get_missing(self, k, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(_args(boot_action="get", key="nonexistent"), k)
        assert exc_info.value.code == 1

    def test_list_empty(self, k, capsys):
        cmd_boot(_args(boot_action="list", format="plain"), k)
        out = capsys.readouterr().out
        assert "(no boot config)" in out

    def test_list_plain(self, k, capsys):
        k.boot_set("model", "claude-3")
        k.boot_set("theme", "dark")
        cmd_boot(_args(boot_action="list", format="plain"), k)
        out = capsys.readouterr().out
        assert "model: claude-3" in out
        assert "theme: dark" in out

    def test_list_json(self, k, capsys):
        k.boot_set("model", "claude-3")
        cmd_boot(_args(boot_action="list", format="json"), k)
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert parsed["model"] == "claude-3"

    def test_list_md(self, k, capsys):
        k.boot_set("model", "claude-3")
        cmd_boot(_args(boot_action="list", format="md"), k)
        out = capsys.readouterr().out
        assert "## Boot Config" in out
        assert "- model: claude-3" in out

    def test_delete_existing(self, k, capsys):
        k.boot_set("model", "claude-3")
        cmd_boot(_args(boot_action="delete", key="model"), k)
        out = capsys.readouterr().out
        assert "Deleted: model" in out

    def test_delete_missing(self, k, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(_args(boot_action="delete", key="nonexistent"), k)
        assert exc_info.value.code == 1

    def test_clear_without_confirm(self, k, capsys):
        with pytest.raises(SystemExit) as exc_info:
            cmd_boot(_args(boot_action="clear", confirm=False), k)
        assert exc_info.value.code == 2

    def test_clear_with_confirm(self, k, capsys):
        k.boot_set("a", "1")
        k.boot_set("b", "2")
        cmd_boot(_args(boot_action="clear", confirm=True), k)
        out = capsys.readouterr().out
        assert "Cleared 2 boot config entries" in out

    def test_export_with_config(self, k, capsys):
        k.boot_set("model", "claude-3")
        cmd_boot(_args(boot_action="export", output=None), k)
        out = capsys.readouterr().out
        assert "Boot config exported to" in out

    def test_export_to_custom_path(self, k, capsys, tmp_path):
        k.boot_set("model", "claude-3")
        output = str(tmp_path / "boot_export.md")
        cmd_boot(_args(boot_action="export", output=output), k)
        out = capsys.readouterr().out
        assert f"Boot config exported to {output}" in out
        assert Path(output).exists()

    def test_export_empty_config(self, k, capsys, tmp_path):
        output = str(tmp_path / "boot_export.md")
        cmd_boot(_args(boot_action="export", output=output), k)
        out = capsys.readouterr().out
        assert "(no boot config to export)" in out

    def test_no_action(self, k, capsys):
        """cmd_boot with no boot_action does nothing (no crash)."""
        cmd_boot(_args(boot_action=None), k)
        out = capsys.readouterr().out
        assert out == ""


# ===========================================================================
# cmd_drive
# ===========================================================================


class TestCmdDrive:
    def test_list_empty(self, k, capsys):
        cmd_drive(_args(drive_action="list"), k)
        out = capsys.readouterr().out
        assert "No drives set" in out

    def test_set_and_list(self, k, capsys):
        cmd_drive(
            _args(drive_action="set", type="curiosity", intensity=0.7, focus=["ai", "ml"]),
            k,
        )
        out = capsys.readouterr().out
        assert "Drive 'curiosity' set to 70%" in out

        cmd_drive(_args(drive_action="list"), k)
        out = capsys.readouterr().out
        assert "Drives:" in out
        assert "curiosity: 70%" in out
        assert "ai, ml" in out

    def test_list_drive_no_focus(self, k, capsys):
        k.drive("growth", intensity=0.5)
        cmd_drive(_args(drive_action="list"), k)
        out = capsys.readouterr().out
        assert "growth: 50%" in out

    def test_satisfy_existing(self, k, capsys):
        k.drive("curiosity", intensity=0.8)
        cmd_drive(_args(drive_action="satisfy", type="curiosity", amount=0.2), k)
        out = capsys.readouterr().out
        assert "Satisfied drive 'curiosity'" in out

    def test_satisfy_missing(self, k, capsys):
        cmd_drive(_args(drive_action="satisfy", type="nonexistent", amount=0.2), k)
        out = capsys.readouterr().out
        assert "Drive 'nonexistent' not found" in out
