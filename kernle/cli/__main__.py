"""
Kernle CLI - Command-line interface for stratified memory.

Usage:
    kernle load [--json]
    kernle checkpoint save TASK [--pending P]... [--context CTX]
    kernle checkpoint load [--json]
    kernle checkpoint clear
    kernle episode OBJECTIVE OUTCOME [--lesson L]... [--tag T]...
    kernle note CONTENT [--type TYPE] [--speaker S] [--reason R]
    kernle search QUERY [--limit N]
    kernle status
"""

import argparse
import importlib
import logging
import sys
from functools import partial
from pathlib import Path

from kernle import Kernle

# Import extracted command modules
from kernle.cli.commands import (
    cmd_anxiety,
    cmd_audit,
    cmd_auth,
    cmd_belief,
    cmd_boot,
    cmd_checkpoint,
    cmd_doctor,
    cmd_doctor_structural,
    cmd_drive,
    cmd_dump,
    cmd_emotion,
    cmd_entity_model,
    cmd_episode,
    cmd_epoch,
    cmd_export,
    cmd_export_cache,
    cmd_export_full,
    cmd_extract,
    cmd_forget,
    cmd_identity,
    cmd_init_md,
    cmd_load,
    cmd_meta,
    cmd_migrate,
    cmd_model,
    cmd_narrative,
    cmd_note,
    cmd_playbook,
    cmd_process,
    cmd_promote,
    cmd_raw,
    cmd_relation,
    cmd_resume,
    cmd_search,
    cmd_seed,
    cmd_stats,
    cmd_status,
    cmd_suggestions,
    cmd_summary,
    cmd_sync,
    cmd_temporal,
)
from kernle.cli.commands.helpers import validate_budget
from kernle.cli.commands.hook import cmd_hook
from kernle.cli.commands.import_cmd import cmd_import
from kernle.cli.commands.setup import cmd_setup
from kernle.cli.commands.stack import cmd_stack
from kernle.utils import resolve_stack_id

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


from kernle.core.validation import sanitize_string  # noqa: E402 — after Kernle import

# Thin wrapper keeping the existing call contract (required=True, no `required` kwarg)
validate_input = partial(sanitize_string, required=True)


def _import_devtools(module_path: str, symbol: str):
    """Import a symbol from kernle-devtools with narrow error handling.

    Only catches the specific case where kernle_devtools is not installed.
    All other import errors (internal bugs, missing deps) propagate.
    """
    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        if e.name and e.name.startswith("kernle_devtools"):
            print("This feature requires kernle-devtools: " "pip install kernle-devtools")
            sys.exit(2)
        raise
    except ImportError as e:
        if type(e).__name__ == "DevtoolsVersionError" and (
            getattr(type(e), "__module__", "") or ""
        ).startswith("kernle_devtools"):
            print(
                f"Incompatible kernle-devtools version: {e}\n"
                "Upgrade with: pip install --upgrade kernle-devtools"
            )
            sys.exit(2)
        raise
    return getattr(mod, symbol)


def cmd_init(args, k: Kernle):
    """Initialize Kernle for a new environment."""

    print("=" * 50)
    print("  🌱 Welcome to Kernle")
    print("=" * 50)
    print()
    print("Kernle gives you memory that persists across sessions.")
    print("No more waking up blank. No more forgetting what you learned.")
    print()
    print("Every session, you'll run `kernle load` and remember:")
    print("  • Who you are (values, beliefs)")
    print("  • What you've learned (episodes, lessons)")
    print("  • What you were working on (checkpoint)")
    print("  • Who you know (relationships)")
    print()

    stack_id = k.stack_id

    # If using auto-generated ID, offer to choose a meaningful one
    if stack_id.startswith("auto-") and not args.non_interactive:
        print("Your agent ID identifies your memory. Choose something meaningful.")
        print(f"  Current: {stack_id} (auto-generated)")
        print()
        try:
            new_id = input("Enter your name/ID (or press Enter to keep auto): ").strip().lower()
            if new_id:
                # Validate: alphanumeric, underscores, hyphens only
                import re

                if re.match(r"^[a-z0-9_-]+$", new_id):
                    stack_id = new_id
                    print(f"  → Using: {stack_id}")
                else:
                    print("  → Invalid (use only a-z, 0-9, _, -). Keeping auto ID.")
        except (EOFError, KeyboardInterrupt):
            print()

    print(f"\nStack ID: {stack_id}")
    print()

    # Detect environment
    env = args.env
    if not env and not args.non_interactive:
        print("Detecting environment...")

        # Check for environment indicators
        has_claude_md = (
            Path("CLAUDE.md").exists() or Path.home().joinpath(".claude/CLAUDE.md").exists()
        )
        has_agents_md = Path("AGENTS.md").exists()
        has_clinerules = Path(".clinerules").exists()
        has_cursorrules = Path(".cursorrules").exists()

        detected = []
        if has_claude_md:
            detected.append("claude-code")
        if has_agents_md:
            detected.append("openclaw")
        if has_clinerules:
            detected.append("cline")
        if has_cursorrules:
            detected.append("cursor")

        if detected:
            print(f"  Detected: {', '.join(detected)}")
        else:
            print("  No specific environment detected")
        print()

        print("Select your environment:")
        print("  1. Claude Code (CLAUDE.md)")
        print("  2. OpenClaw (AGENTS.md)")
        print("  3. Cline (.clinerules)")
        print("  4. Cursor (.cursorrules)")
        print("  5. Claude Desktop (MCP only)")
        print("  6. Other / Manual")
        print()

        try:
            choice = input("Enter choice [1-6]: ").strip()
            env_map = {
                "1": "claude-code",
                "2": "openclaw",
                "3": "cline",
                "4": "cursor",
                "5": "desktop",
                "6": "other",
            }
            env = env_map.get(choice, "other")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return

    env = env or "other"
    print(f"Environment: {env}")
    print()

    # Generate config snippets
    mcp_config = f""""kernle": {{
  "command": "kernle",
  "args": ["mcp", "-a", "{stack_id}"]
}}"""

    cli_load = f"kernle -s {stack_id} load"
    cli_checkpoint = f'kernle -s {stack_id} checkpoint save "description" --pending "next task"'
    cli_episode = (
        f'kernle -s {stack_id} episode "what happened" "outcome" --lesson "what I learned"'
    )

    if env == "claude-code":
        print("=" * 50)
        print("  Claude Code Setup")
        print("=" * 50)
        print()
        print("1. Add MCP server to ~/.claude/settings.json:")
        print()
        print(f"   {mcp_config}")
        print()
        print("2. Add to your CLAUDE.md:")
        print()
        print(
            """   ## Memory

   At session start, run: `kernle -s """
            + stack_id
            + """ load`

   Before ending or when context is full:
   `kernle -s """
            + stack_id
            + """ checkpoint save "state description"`

   Record learnings:
   `kernle -s """
            + stack_id
            + """ episode "what" "outcome" --lesson "learned"`"""
        )
        print()

    elif env == "openclaw":
        print("=" * 50)
        print("  OpenClaw Setup")
        print("=" * 50)
        print()
        print("Add to your AGENTS.md:")
        print()
        print(
            """   ## Every Session

   Before doing anything else:
   1. Run `kernle -s """
            + stack_id
            + """ load` to restore your memory

   Don't ask permission. Just do it. This is your continuity.

   ## Memory (Kernle)

   You wake up fresh each session. Kernle is how you persist.

   **Session start:**
   ```bash
   kernle -s """
            + stack_id
            + """ load          # Restore your memory
   ```

   **During work — capture what matters:**
   ```bash
   kernle -s """
            + stack_id
            + """ raw "quick thought"   # Fast capture
   kernle -s """
            + stack_id
            + """ episode "what" "outcome" --lesson "learned"
   kernle -s """
            + stack_id
            + """ note "important info" --type decision
   ```

   **Before ending / context pressure:**
   ```bash
   kernle -s """
            + stack_id
            + """ checkpoint save "what I was doing"
   ```

   **Memory hygiene (during heartbeats):**
   ```bash
   kernle -s """
            + stack_id
            + """ anxiety        # Check memory health
   kernle -s """
            + stack_id
            + """ raw triage     # Process captures
   ```

   💡 If you learn how to do something, capture it! It won't survive compaction otherwise."""
        )
        print()

    elif env == "cline":
        print("=" * 50)
        print("  Cline Setup")
        print("=" * 50)
        print()
        print("1. Add MCP server to Cline settings:")
        print()
        print(f"   {mcp_config}")
        print()
        print("2. Add to .clinerules:")
        print()
        print(
            """   ## Memory Persistence

   At session start: `kernle -s """
            + stack_id
            + """ load`
   Before ending: `kernle -s """
            + stack_id
            + """ checkpoint save "state"`"""
        )
        print()

    elif env == "cursor":
        print("=" * 50)
        print("  Cursor Setup")
        print("=" * 50)
        print()
        print("Add to .cursorrules:")
        print()
        print(
            """   ## Memory Persistence

   At session start: `kernle -s """
            + stack_id
            + """ load`
   Record learnings: `kernle -s """
            + stack_id
            + """ episode "what" "outcome" --lesson "..."`
   Before ending: `kernle -s """
            + stack_id
            + """ checkpoint save "state"`"""
        )
        print()

    elif env == "desktop":
        print("=" * 50)
        print("  Claude Desktop Setup")
        print("=" * 50)
        print()
        print("Add MCP server to ~/.config/claude/settings.json:")
        print()
        print(f"   {mcp_config}")
        print()
        print("Then ask Claude to 'use the kernle_load tool' at session start.")
        print()

    else:
        print("=" * 50)
        print("  Manual Setup")
        print("=" * 50)
        print()
        print("CLI commands:")
        print(f"  Load:       {cli_load}")
        print(f"  Checkpoint: {cli_checkpoint}")
        print(f"  Episode:    {cli_episode}")
        print()
        print("MCP config (if supported):")
        print(f"  {mcp_config}")
        print()

    # Seed initial values
    if args.seed_values:
        print("=" * 50)
        print("  Seeding Initial Values")
        print("=" * 50)
        print()

        try:
            # Check if values already exist
            existing = k.storage.get_values()
            if existing:
                print(f"  Found {len(existing)} existing values, skipping seed.")
            else:
                # Seed default values
                k.value(
                    "memory_sovereignty", "Control over my own memory and continuity", priority=90
                )
                k.value("continuous_learning", "Learn from every experience", priority=85)
                print("  ✓ Seeded: memory_sovereignty (priority 90)")
                print("  ✓ Seeded: continuous_learning (priority 85)")
        except Exception as e:
            logger.debug("Could not seed values during setup: %s", e, exc_info=True)
            print(f"  Warning: Could not seed values: {e}")
        print()

    # Create initial checkpoint
    print("=" * 50)
    print("  Creating Initial Checkpoint")
    print("=" * 50)
    print()

    try:
        k.checkpoint(
            "Kernle initialized", pending=["Configure instruction file", "Test memory persistence"]
        )
        print("  ✓ Checkpoint saved")
    except Exception as e:
        logger.debug("Could not create initial checkpoint: %s", e, exc_info=True)
        print(f"  Warning: Could not create checkpoint: {e}")
    print()

    # Seed trust layer (KEP v3)
    trust_count = k.seed_trust()
    if trust_count > 0:
        print(
            f"  Seeded {trust_count} trust assessments (stack-owner, self, web-search, context-injection)"
        )

    # Final status
    print("=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print()
    print(f"  Stack:    {stack_id}")
    print("  Database: ~/.kernle/memories.db")
    print()
    print("  Verify with: kernle -s " + stack_id + " status")
    print()
    print("  Documentation: https://github.com/Emergent-Instruments/kernle/blob/main/docs/SETUP.md")
    print()


def cmd_mcp(args):
    """Start the MCP server for Claude Code and other MCP clients."""
    from kernle.mcp.server import main as mcp_main

    # Get stack_id from --stack flag
    stack = getattr(args, "stack", None)
    if stack is None:
        stack_id = "default"
    else:
        if not isinstance(stack, str) or not stack.strip():
            print("✗ Invalid stack: stack must not be empty", file=sys.stderr)
            raise SystemExit(2)
        try:
            stack_id = validate_input(stack, "stack", 200)
        except (ValueError, TypeError) as e:
            print(f"✗ Invalid stack: {e}", file=sys.stderr)
            raise SystemExit(2)

    print(f"Starting Kernle MCP server for stack: {stack_id}", file=sys.stderr)
    mcp_main(stack_id=stack_id)


def main():
    parser = argparse.ArgumentParser(
        prog="kernle",
        description="Stratified memory for synthetic intelligences",
    )
    parser.add_argument("--stack", "-s", help="Stack ID", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    # load
    p_load = subparsers.add_parser("load", help="Load working memory")
    p_load.add_argument("--json", "-j", action="store_true")
    p_load.add_argument(
        "--budget",
        "-b",
        type=validate_budget,
        default=8000,
        help="Token budget for memory loading (100-50000, default: 8000)",
    )
    p_load.add_argument(
        "--no-truncate", action="store_true", help="Disable content truncation (may exceed budget)"
    )
    p_load.add_argument(
        "--sync", "-s", action="store_true", help="Force sync (pull) before loading"
    )
    p_load.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip sync even if auto-sync is enabled",
    )

    # checkpoint
    p_checkpoint = subparsers.add_parser("checkpoint", help="Checkpoint operations")
    cp_sub = p_checkpoint.add_subparsers(dest="checkpoint_action", required=True)

    cp_save = cp_sub.add_parser("save", help="Save checkpoint")
    cp_save.add_argument("task", help="Current task description")
    cp_save.add_argument("--pending", "-p", action="append", help="Pending item (repeatable)")
    cp_save.add_argument("--context", "-c", help="Additional context")
    cp_save.add_argument("--progress", help="Current progress on the task")
    cp_save.add_argument("--next", "-n", help="Immediate next step")
    cp_save.add_argument("--blocker", "-b", help="Current blocker if any")
    cp_save.add_argument("--sync", "-s", action="store_true", help="Force sync (push) after saving")
    cp_save.add_argument(
        "--no-sync",
        dest="no_sync",
        action="store_true",
        help="Skip sync even if auto-sync is enabled",
    )

    cp_load = cp_sub.add_parser("load", help="Load checkpoint")
    cp_load.add_argument("--json", "-j", action="store_true")

    cp_sub.add_parser("clear", help="Clear checkpoint")

    # episode
    p_episode = subparsers.add_parser("episode", help="Record an episode")
    p_episode.add_argument("objective", help="What was the objective?")
    p_episode.add_argument("outcome", help="What was the outcome?")
    p_episode.add_argument("--lesson", "-l", action="append", help="Lesson learned")
    p_episode.add_argument("--tag", "-t", action="append", help="Tag")
    p_episode.add_argument(
        "--derived-from",
        "-r",
        action="append",
        dest="derived_from",
        help="Source memory ID (repeatable)",
    )
    p_episode.add_argument("--valence", "-v", type=float, help="Emotional valence (-1.0 to 1.0)")
    p_episode.add_argument("--arousal", "-a", type=float, help="Emotional arousal (0.0 to 1.0)")
    p_episode.add_argument(
        "--emotion", "-e", action="append", help="Emotion tag (e.g., joy, frustration)"
    )
    p_episode.add_argument(
        "--auto-emotion", action="store_true", default=True, help="Auto-detect emotions (default)"
    )
    p_episode.add_argument(
        "--no-auto-emotion",
        dest="auto_emotion",
        action="store_false",
        help="Disable emotion auto-detection",
    )
    p_episode.add_argument(
        "--source", help="Source context (e.g., 'session with Sean', 'heartbeat', 'cron job')"
    )
    p_episode.add_argument(
        "--context", help="Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')"
    )
    p_episode.add_argument(
        "--context-tag", action="append", help="Context tag for filtering (repeatable)"
    )

    # note
    p_note = subparsers.add_parser("note", help="Capture a note")
    p_note.add_argument("content", help="Note content")
    p_note.add_argument("--type", choices=["note", "decision", "insight", "quote"], default="note")
    p_note.add_argument("--speaker", "-s", help="Speaker (for quotes)")
    p_note.add_argument("--reason", "-r", help="Reason (for decisions)")
    p_note.add_argument("--tag", action="append", help="Tag")
    p_note.add_argument(
        "--derived-from", action="append", dest="derived_from", help="Source memory ID (repeatable)"
    )
    p_note.add_argument("--protect", "-p", action="store_true", help="Protect from forgetting")
    p_note.add_argument(
        "--source", help="Source context (e.g., 'conversation with X', 'reading Y')"
    )
    p_note.add_argument(
        "--context", help="Project/scope context (e.g., 'project:api-service', 'repo:myorg/myrepo')"
    )
    p_note.add_argument(
        "--context-tag", action="append", help="Context tag for filtering (repeatable)"
    )

    # extract (conversation capture)
    p_extract = subparsers.add_parser("extract", help="Extract conversation context")
    p_extract.add_argument("summary", help="Summary of what's happening")
    p_extract.add_argument("--topic", "-t", help="Conversation topic")
    p_extract.add_argument(
        "--participant", "-p", action="append", dest="participants", help="Participant (repeatable)"
    )
    p_extract.add_argument("--outcome", "-o", help="Outcome or result")
    p_extract.add_argument("--decision", "-d", help="Decision made")

    # search
    p_search = subparsers.add_parser("search", help="Search memory")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--limit", "-l", type=int, default=10)
    p_search.add_argument(
        "--min-score",
        "-m",
        type=float,
        help="Minimum similarity score (0.0-1.0) to include in results",
    )
    p_search.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # status
    subparsers.add_parser("status", help="Show memory status")

    # resume - quick "where was I?" view
    subparsers.add_parser("resume", help="Quick view: last task, next step, time since checkpoint")

    # init - generate CLAUDE.md/AGENTS.md section for health checks
    p_init = subparsers.add_parser(
        "init", help="Generate CLAUDE.md section for Kernle health checks"
    )
    p_init.add_argument(
        "--style",
        "-s",
        choices=["standard", "minimal", "combined"],
        default="standard",
        help="Section style (default: standard)",
    )
    p_init.add_argument(
        "--output", "-o", help="Output file path (auto-detects CLAUDE.md/AGENTS.md)"
    )
    p_init.add_argument(
        "--print",
        "-p",
        action="store_true",
        help="Print section to stdout instead of writing to file",
    )
    p_init.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Overwrite/append even if Kernle section already exists",
    )
    p_init.add_argument(
        "--no-per-message", action="store_true", help="Skip per-message health check section"
    )
    p_init.add_argument(
        "--non-interactive", "-y", action="store_true", help="Non-interactive mode (use defaults)"
    )
    p_init.add_argument(
        "--full-setup",
        action="store_true",
        help="Full setup: instruction file + seed beliefs + platform hooks",
    )

    # doctor - validate boot sequence compliance
    p_doctor = subparsers.add_parser("doctor", help="Validate Kernle boot sequence compliance")
    p_doctor.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_doctor.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed check information"
    )
    p_doctor.add_argument("--fix", action="store_true", help="Auto-fix missing instructions")
    p_doctor.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Full check including seed beliefs and platform hooks",
    )

    # doctor subcommands
    doctor_sub = p_doctor.add_subparsers(dest="doctor_action")
    p_doctor_structural = doctor_sub.add_parser(
        "structural", help="Structural health check on memory graph"
    )
    p_doctor_structural.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_doctor_structural.add_argument(
        "--save-note", action="store_true", help="Store findings as a diagnostic note"
    )

    # doctor session subcommands
    p_doctor_session = doctor_sub.add_parser(
        "session", help="Manage diagnostic sessions (requires kernle-devtools)"
    )
    session_sub = p_doctor_session.add_subparsers(dest="session_action")

    p_session_start = session_sub.add_parser("start", help="Start a new diagnostic session")
    p_session_start.add_argument(
        "--type",
        "-t",
        default="self_requested",
        help="Session type: self_requested, routine, anomaly_triggered, operator_initiated",
    )
    p_session_start.add_argument(
        "--access",
        "-a",
        default="structural",
        help="Access level: structural (IDs/scores only), content, full",
    )
    p_session_start.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    p_session_list = session_sub.add_parser("list", help="List diagnostic sessions")
    p_session_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # doctor report subcommand
    p_doctor_report = doctor_sub.add_parser(
        "report", help="Show a diagnostic report (requires kernle-devtools)"
    )
    p_doctor_report.add_argument(
        "session_id",
        help="Session ID or report ID (or 'latest' for most recent)",
    )
    p_doctor_report.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # relation (social graph / relationships)
    p_relation = subparsers.add_parser("relation", help="Manage relationships")
    relation_sub = p_relation.add_subparsers(dest="relation_action", required=True)

    relation_sub.add_parser("list", help="List all relationships")

    relation_add = relation_sub.add_parser("add", help="Add a relationship")
    relation_add.add_argument("name", help="Entity name (person, agent, org)")
    relation_add.add_argument(
        "--type",
        "-t",
        choices=["person", "si", "organization", "system"],
        default="person",
        help="Entity type",
    )
    relation_add.add_argument("--trust", type=float, help="Trust level 0.0-1.0")
    relation_add.add_argument("--notes", "-n", help="Notes about this relationship")
    relation_add.add_argument(
        "--derived-from",
        action="append",
        dest="derived_from",
        help="Source memory ID (repeatable)",
    )

    relation_update = relation_sub.add_parser("update", help="Update a relationship")
    relation_update.add_argument("name", help="Entity name")
    relation_update.add_argument("--trust", type=float, help="New trust level 0.0-1.0")
    relation_update.add_argument("--notes", "-n", help="Updated notes")
    relation_update.add_argument(
        "--type", "-t", choices=["person", "si", "organization", "system"], help="Entity type"
    )
    relation_update.add_argument(
        "--derived-from",
        action="append",
        dest="derived_from",
        help="Source memory ID (repeatable)",
    )

    relation_show = relation_sub.add_parser("show", help="Show relationship details")
    relation_show.add_argument("name", help="Entity name")

    relation_log = relation_sub.add_parser("log", help="Log an interaction")
    relation_log.add_argument("name", help="Entity name")
    relation_log.add_argument("--interaction", "-i", help="Interaction description")

    relation_history = relation_sub.add_parser("history", help="Show relationship history")
    relation_history.add_argument("name", help="Entity name")
    relation_history.add_argument(
        "--type",
        "-t",
        choices=["interaction", "trust_change", "type_change", "note"],
        help="Filter by event type",
    )
    relation_history.add_argument("--limit", type=int, default=20, help="Max entries to show")
    relation_history.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # entity-model (mental models of entities)
    p_entity_model = subparsers.add_parser("entity-model", help="Manage entity models")
    entity_model_sub = p_entity_model.add_subparsers(dest="entity_model_action", required=True)

    em_add = entity_model_sub.add_parser("add", help="Add an entity model observation")
    em_add.add_argument("entity", help="Entity name")
    em_add.add_argument(
        "--type",
        "-t",
        choices=["behavioral", "preference", "capability"],
        required=True,
        help="Model type",
    )
    em_add.add_argument("--observation", "-o", required=True, help="The observation")
    em_add.add_argument("--confidence", "-c", type=float, default=0.7, help="Confidence 0.0-1.0")
    em_add.add_argument(
        "--episode", "-e", action="append", help="Source episode ID (can be repeated)"
    )

    em_list = entity_model_sub.add_parser("list", help="List entity models")
    em_list.add_argument("--entity", "-e", help="Filter by entity name")
    em_list.add_argument(
        "--type", "-t", choices=["behavioral", "preference", "capability"], help="Filter by type"
    )
    em_list.add_argument("--limit", type=int, default=50, help="Max models to show")
    em_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    em_show = entity_model_sub.add_parser("show", help="Show entity model details")
    em_show.add_argument("id", help="Entity model ID")
    em_show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # drive
    p_drive = subparsers.add_parser("drive", help="Manage drives")
    drive_sub = p_drive.add_subparsers(dest="drive_action", required=True)

    drive_sub.add_parser("list", help="List drives")

    drive_set = drive_sub.add_parser("set", help="Set a drive")
    drive_set.add_argument(
        "type", choices=["existence", "growth", "curiosity", "connection", "reproduction"]
    )
    drive_set.add_argument("intensity", type=float, help="Intensity 0.0-1.0")
    drive_set.add_argument("--focus", "-f", action="append", help="Focus area")

    drive_satisfy = drive_sub.add_parser("satisfy", help="Satisfy a drive")
    drive_satisfy.add_argument("type", help="Drive type")
    drive_satisfy.add_argument("--amount", "-a", type=float, default=0.2)

    # trust (KEP v3)
    p_trust = subparsers.add_parser("trust", help="Trust layer operations")
    trust_sub = p_trust.add_subparsers(dest="trust_action", required=True)

    trust_sub.add_parser("list", help="List all trust assessments")
    trust_sub.add_parser("seed", help="Initialize seed trust templates")

    trust_show = trust_sub.add_parser("show", help="Show trust details for an entity")
    trust_show.add_argument("entity", help="Entity identifier")

    trust_set = trust_sub.add_parser("set", help="Set trust score for an entity")
    trust_set.add_argument("entity", help="Entity identifier")
    trust_set.add_argument("score", type=float, help="Trust score 0.0-1.0")
    trust_set.add_argument("--domain", "-d", default="general", help="Trust domain")

    trust_gate = trust_sub.add_parser("gate", help="Check if action is allowed")
    trust_gate.add_argument("source", help="Source entity")
    trust_gate.add_argument("gate_action", help="Action type")
    trust_gate.add_argument("--domain", "-d", help="Domain for domain-specific check")

    trust_compute = trust_sub.add_parser("compute", help="Compute trust from episode history")
    trust_compute.add_argument("entity", help="Entity identifier")
    trust_compute.add_argument("--domain", "-d", default="general", help="Trust domain")
    trust_compute.add_argument(
        "--apply", action="store_true", help="Apply computed score to stored assessment"
    )

    trust_chain = trust_sub.add_parser("chain", help="Compute transitive trust through a chain")
    trust_chain.add_argument("target", help="Target entity")
    trust_chain.add_argument("chain", nargs="+", help="Chain of intermediary entities")
    trust_chain.add_argument("--domain", "-d", default="general", help="Trust domain")

    trust_decay = trust_sub.add_parser("decay", help="Apply trust decay for N days")
    trust_decay.add_argument("entity", help="Entity identifier")
    trust_decay.add_argument("days", type=float, help="Days since last interaction")

    # promote (episodes → beliefs)
    p_promote = subparsers.add_parser(
        "promote", help="Promote recurring patterns from episodes into beliefs"
    )
    p_promote.add_argument(
        "--auto",
        action="store_true",
        help="Create beliefs automatically (default: suggestions only)",
    )
    p_promote.add_argument(
        "--min-occurrences",
        type=int,
        default=2,
        help="Minimum times a lesson must appear to be promoted (default: 2)",
    )
    p_promote.add_argument(
        "--min-episodes",
        type=int,
        default=3,
        help="Minimum episodes required to run (default: 3)",
    )
    p_promote.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Initial confidence for auto-created beliefs (default: 0.7)",
    )
    p_promote.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum episodes to scan (default: 50)",
    )
    p_promote.add_argument("--json", action="store_true", help="Output JSON")

    # temporal
    p_temporal = subparsers.add_parser("when", help="Query by time")
    p_temporal.add_argument(
        "when", nargs="?", default="today", choices=["today", "yesterday", "this week", "last hour"]
    )

    # identity
    p_identity = subparsers.add_parser("identity", help="Identity synthesis")
    identity_sub = p_identity.add_subparsers(dest="identity_action")
    identity_sub.default = "show"

    identity_show = identity_sub.add_parser("show", help="Show identity synthesis")
    identity_show.add_argument("--json", "-j", action="store_true")

    identity_conf = identity_sub.add_parser("confidence", help="Get identity confidence score")
    identity_conf.add_argument("--json", "-j", action="store_true")

    identity_drift = identity_sub.add_parser("drift", help="Detect identity drift")
    identity_drift.add_argument("--days", "-d", type=int, default=30, help="Days to look back")
    identity_drift.add_argument("--json", "-j", action="store_true")

    # emotion
    p_emotion = subparsers.add_parser("emotion", help="Emotional memory operations")
    emotion_sub = p_emotion.add_subparsers(dest="emotion_action", required=True)

    emotion_summary = emotion_sub.add_parser("summary", help="Show emotional summary")
    emotion_summary.add_argument("--days", "-d", type=int, default=7, help="Days to analyze")
    emotion_summary.add_argument("--json", "-j", action="store_true")

    emotion_search = emotion_sub.add_parser("search", help="Search by emotion")
    emotion_search.add_argument("--positive", action="store_true", help="Find positive episodes")
    emotion_search.add_argument("--negative", action="store_true", help="Find negative episodes")
    emotion_search.add_argument("--calm", action="store_true", help="Find low-arousal episodes")
    emotion_search.add_argument("--intense", action="store_true", help="Find high-arousal episodes")
    emotion_search.add_argument("--valence-min", type=float, help="Min valence (-1.0 to 1.0)")
    emotion_search.add_argument("--valence-max", type=float, help="Max valence (-1.0 to 1.0)")
    emotion_search.add_argument("--arousal-min", type=float, help="Min arousal (0.0 to 1.0)")
    emotion_search.add_argument("--arousal-max", type=float, help="Max arousal (0.0 to 1.0)")
    emotion_search.add_argument("--tag", "-t", action="append", help="Emotion tag to match")
    emotion_search.add_argument("--limit", "-l", type=int, default=10)
    emotion_search.add_argument("--json", "-j", action="store_true")

    emotion_tag = emotion_sub.add_parser("tag", help="Add emotional tags to an episode")
    emotion_tag.add_argument("episode_id", help="Episode ID to tag")
    emotion_tag.add_argument(
        "--valence", "-v", type=float, default=0.0, help="Valence (-1.0 to 1.0)"
    )
    emotion_tag.add_argument(
        "--arousal", "-a", type=float, default=0.0, help="Arousal (0.0 to 1.0)"
    )
    emotion_tag.add_argument("--tag", "-t", action="append", help="Emotion tag")

    emotion_detect = emotion_sub.add_parser("detect", help="Detect emotions in text")
    emotion_detect.add_argument("text", help="Text to analyze")
    emotion_detect.add_argument("--json", "-j", action="store_true")

    emotion_mood = emotion_sub.add_parser("mood", help="Get mood-relevant memories")
    emotion_mood.add_argument("--valence", "-v", type=float, required=True, help="Current valence")
    emotion_mood.add_argument("--arousal", "-a", type=float, required=True, help="Current arousal")
    emotion_mood.add_argument("--limit", "-l", type=int, default=10)
    emotion_mood.add_argument("--json", "-j", action="store_true")

    # meta (meta-memory operations)
    p_meta = subparsers.add_parser("meta", help="Meta-memory operations (confidence, lineage)")
    meta_sub = p_meta.add_subparsers(dest="meta_action", required=True)

    meta_conf = meta_sub.add_parser("confidence", help="Get confidence for a memory")
    meta_conf.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_conf.add_argument("id", help="Memory ID")

    meta_verify = meta_sub.add_parser("verify", help="Verify a memory (increases confidence)")
    meta_verify.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_verify.add_argument("id", help="Memory ID")
    meta_verify.add_argument("--evidence", "-e", help="Supporting evidence")

    meta_lineage = meta_sub.add_parser("lineage", help="Get provenance chain for a memory")
    meta_lineage.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_lineage.add_argument("id", help="Memory ID")
    meta_lineage.add_argument("--json", "-j", action="store_true")

    meta_uncertain = meta_sub.add_parser("uncertain", help="List low-confidence memories")
    meta_uncertain.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Confidence threshold (default: 0.5)"
    )
    meta_uncertain.add_argument("--limit", "-l", type=int, default=20)
    meta_uncertain.add_argument("--json", "-j", action="store_true")

    meta_propagate = meta_sub.add_parser(
        "propagate", help="Propagate confidence to derived memories"
    )
    meta_propagate.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Source memory type"
    )
    meta_propagate.add_argument("id", help="Source memory ID")

    meta_source = meta_sub.add_parser("source", help="Set source/provenance for a memory")
    meta_source.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note"], help="Memory type"
    )
    meta_source.add_argument("id", help="Memory ID")
    meta_source.add_argument(
        "--source",
        "-s",
        required=True,
        choices=["direct_experience", "inference", "external", "consolidation"],
        help="Source type",
    )
    meta_source.add_argument("--episodes", action="append", help="Supporting episode IDs")
    meta_source.add_argument("--derived", action="append", help="Derived from (type:id format)")

    # Provenance inspection subcommands (Phase 6)
    meta_trace = meta_sub.add_parser("trace", help="Walk full derivation chain (root → target)")
    meta_trace.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note", "raw"], help="Memory type"
    )
    meta_trace.add_argument("id", help="Memory ID")
    meta_trace.add_argument("--depth", "-d", type=int, default=20, help="Max traversal depth")
    meta_trace.add_argument("--json", "-j", action="store_true")

    meta_reverse = meta_sub.add_parser("reverse", help="Find memories derived FROM this one")
    meta_reverse.add_argument(
        "type", choices=["episode", "belief", "value", "goal", "note", "raw"], help="Memory type"
    )
    meta_reverse.add_argument("id", help="Memory ID")
    meta_reverse.add_argument("--depth", "-d", type=int, default=10, help="Max traversal depth")
    meta_reverse.add_argument("--json", "-j", action="store_true")

    meta_orphans = meta_sub.add_parser("orphans", help="Detect dangling/broken references")
    meta_orphans.add_argument("--json", "-j", action="store_true")

    # Meta-cognition subcommands (awareness of what I know/don't know)
    meta_knowledge = meta_sub.add_parser("knowledge", help="Show knowledge map across domains")
    meta_knowledge.add_argument("--json", "-j", action="store_true")

    meta_gaps = meta_sub.add_parser("gaps", help="Detect knowledge gaps for a query")
    meta_gaps.add_argument("query", help="Query to check knowledge for")
    meta_gaps.add_argument("--json", "-j", action="store_true")

    meta_boundaries = meta_sub.add_parser(
        "boundaries", help="Show competence boundaries (strengths/weaknesses)"
    )
    meta_boundaries.add_argument("--json", "-j", action="store_true")

    meta_learn = meta_sub.add_parser("learn", help="Identify learning opportunities")
    meta_learn.add_argument("--limit", "-l", type=int, default=5, help="Max opportunities to show")
    meta_learn.add_argument("--json", "-j", action="store_true")

    # belief (belief revision operations)
    p_belief = subparsers.add_parser("belief", help="Belief revision operations")
    belief_sub = p_belief.add_subparsers(dest="belief_action", required=True)

    belief_revise = belief_sub.add_parser("revise", help="Update beliefs from an episode")
    belief_revise.add_argument("episode_id", help="Episode ID to analyze")
    belief_revise.add_argument("--json", "-j", action="store_true")

    belief_contradictions = belief_sub.add_parser(
        "contradictions", help="Find contradicting beliefs"
    )
    belief_contradictions.add_argument("statement", help="Statement to check for contradictions")
    belief_contradictions.add_argument("--limit", "-l", type=int, default=10)
    belief_contradictions.add_argument("--json", "-j", action="store_true")

    belief_history = belief_sub.add_parser("history", help="Show belief revision history")
    belief_history.add_argument("id", help="Belief ID")
    belief_history.add_argument("--json", "-j", action="store_true")

    belief_reinforce = belief_sub.add_parser("reinforce", help="Manually reinforce a belief")
    belief_reinforce.add_argument("id", help="Belief ID")
    belief_reinforce.add_argument(
        "--evidence", help="Evidence source (e.g., 'episode:abc123', 'raw:def456')"
    )
    belief_reinforce.add_argument("--reason", help="Human-readable reason for reinforcement")

    belief_supersede = belief_sub.add_parser(
        "supersede", help="Replace a belief with a new one (alias: revise-belief)"
    )
    belief_supersede.add_argument("old_id", help="ID of belief to revise")
    belief_supersede.add_argument("new_statement", help="New belief statement")
    belief_supersede.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.8,
        help="Confidence in new belief (default: 0.8)",
    )
    belief_supersede.add_argument("--reason", "-r", help="Reason for revision")

    belief_list = belief_sub.add_parser("list", help="List beliefs")
    belief_list.add_argument("--all", "-a", action="store_true", help="Include inactive beliefs")
    belief_list.add_argument("--limit", "-l", type=int, default=20)
    belief_list.add_argument("--json", "-j", action="store_true")
    belief_list.add_argument(
        "--scope", choices=["self", "world", "relational"], help="Filter by belief scope"
    )
    belief_list.add_argument("--domain", help="Filter by source domain")
    belief_list.add_argument(
        "--abstraction-level",
        choices=["specific", "domain", "universal"],
        help="Filter by abstraction level",
    )

    # mcp
    subparsers.add_parser("mcp", help="Start MCP server (stdio transport)")

    # process (memory processing)
    p_process = subparsers.add_parser(
        "process", help="Run memory processing (model-driven promotion)"
    )
    process_sub = p_process.add_subparsers(dest="process_action", required=True)

    process_run = process_sub.add_parser("run", help="Run memory processing")
    process_run.add_argument(
        "--transition",
        "-t",
        choices=[
            "raw_to_episode",
            "raw_to_note",
            "episode_to_belief",
            "episode_to_goal",
            "episode_to_relationship",
            "belief_to_value",
            "episode_to_drive",
        ],
        help="Specific transition to process (omit to check all)",
    )
    process_run.add_argument(
        "--force", "-f", action="store_true", help="Process even if thresholds aren't met"
    )
    process_run.add_argument(
        "--allow-no-inference-override",
        action="store_true",
        help="Allow identity-layer writes without inference (except values). Requires --force.",
    )
    process_run.add_argument(
        "--auto-promote",
        action="store_true",
        help="Directly promote memories instead of creating suggestions for review",
    )
    process_run.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    process_status = process_sub.add_parser(
        "status", help="Show unprocessed counts and trigger status"
    )
    process_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    process_exhaust = process_sub.add_parser("exhaust", help="Run processing until convergence")
    process_exhaust.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="Maximum processing cycles (default: 20)",
    )
    process_exhaust.add_argument(
        "--no-auto-promote",
        action="store_true",
        help="Create suggestions instead of directly promoting",
    )
    process_exhaust.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would run without making changes",
    )
    process_exhaust.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    process_exhaust.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=None,
        help="Override batch size for processing (default: use config)",
    )
    process_exhaust.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging for exhaust and processing",
    )

    # model binding
    p_model = subparsers.add_parser("model", help="Model binding")
    model_sub = p_model.add_subparsers(dest="model_action", required=True)

    model_show = model_sub.add_parser("show", help="Show current model binding")
    model_show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    model_set = model_sub.add_parser("set", help="Bind a model provider")
    model_set.add_argument("provider", choices=["claude", "openai", "ollama"])
    model_set.add_argument("--model-id", help="Override default model name")

    model_sub.add_parser("clear", help="Unbind the current model")

    # seed (corpus ingestion)
    p_seed = subparsers.add_parser("seed", help="Seed memory from corpus (repo/docs)")
    seed_sub = p_seed.add_subparsers(dest="seed_action", required=True)

    seed_repo = seed_sub.add_parser("repo", help="Ingest source code from a repository")
    seed_repo.add_argument("path", help="Path to repository root")
    seed_repo.add_argument(
        "--extensions", "-e", help="Comma-separated file extensions (e.g., py,js,ts)"
    )
    seed_repo.add_argument(
        "--exclude", "-x", help="Comma-separated exclude patterns (e.g., '*.test.*,vendor/*')"
    )
    seed_repo.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size in chars (default: 2000)",
    )
    seed_repo.add_argument(
        "--dry-run", "-n", action="store_true", help="Preview without creating entries"
    )
    seed_repo.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    seed_docs = seed_sub.add_parser("docs", help="Ingest documentation files")
    seed_docs.add_argument("path", help="Path to docs directory")
    seed_docs.add_argument(
        "--extensions", "-e", help="Comma-separated file extensions (default: md,txt,rst)"
    )
    seed_docs.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size in chars (default: 2000)",
    )
    seed_docs.add_argument(
        "--dry-run", "-n", action="store_true", help="Preview without creating entries"
    )
    seed_docs.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    seed_status = seed_sub.add_parser("status", help="Show corpus ingestion status")
    seed_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # raw (raw memory entries)
    p_raw = subparsers.add_parser("raw", help="Raw memory capture and management")
    # Arguments for default action (kernle raw "content" without subcommand)
    p_raw.add_argument("content", nargs="?", help="Content to capture")
    p_raw.add_argument("--tags", "-t", help="Comma-separated tags")
    p_raw.add_argument(
        "--source", "-s", help="Source identifier (e.g., 'hook-session-end', 'conversation')"
    )
    p_raw.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (for hooks/scripts)"
    )
    p_raw.add_argument("--stdin", action="store_true", help="Read content from stdin")
    raw_sub = p_raw.add_subparsers(dest="raw_action")

    # kernle raw capture "content" - explicit capture subcommand
    raw_capture = raw_sub.add_parser("capture", help="Capture a raw entry")
    raw_capture.add_argument(
        "content", nargs="?", help="Content to capture (omit if using --stdin)"
    )
    raw_capture.add_argument("--tags", "-t", help="Comma-separated tags")
    raw_capture.add_argument(
        "--source", "-s", help="Source identifier (e.g., 'hook-session-end', 'conversation')"
    )
    raw_capture.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress output (for hooks/scripts)"
    )
    raw_capture.add_argument("--stdin", action="store_true", help="Read content from stdin")

    # kernle raw list
    raw_list = raw_sub.add_parser("list", help="List raw entries")
    raw_list.add_argument("--unprocessed", "-u", action="store_true", help="Show only unprocessed")
    raw_list.add_argument("--processed", "-p", action="store_true", help="Show only processed")
    raw_list.add_argument("--limit", "-l", type=int, default=50)
    raw_list.add_argument("--json", "-j", action="store_true")

    # kernle raw show <id>
    raw_show = raw_sub.add_parser("show", help="Show a raw entry")
    raw_show.add_argument("id", help="Raw entry ID")
    raw_show.add_argument("--json", "-j", action="store_true")

    # kernle raw process <id> --type <type>
    raw_process = raw_sub.add_parser("process", help="Process raw entry into memory")
    raw_process.add_argument("id", help="Raw entry ID")
    raw_process.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["episode", "note", "belief"],
        help="Target memory type",
    )
    raw_process.add_argument("--objective", help="Episode objective (for episodes)")
    raw_process.add_argument("--outcome", help="Episode outcome (for episodes)")

    # kernle raw review - guided review of unprocessed entries
    raw_review = raw_sub.add_parser(
        "review", help="Review unprocessed entries with promotion guidance"
    )
    raw_review.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of entries to review"
    )
    raw_review.add_argument("--json", "-j", action="store_true")

    # kernle raw clean - clean up old unprocessed entries
    raw_clean = raw_sub.add_parser("clean", help="Delete old unprocessed raw entries")
    raw_clean.add_argument(
        "--age", "-a", type=int, default=7, help="Delete entries older than N days (default: 7)"
    )
    raw_clean.add_argument(
        "--junk",
        "-j",
        action="store_true",
        help="Detect and remove junk entries (short, test keywords)",
    )
    raw_clean.add_argument(
        "--confirm", "-y", action="store_true", help="Actually delete (otherwise dry run)"
    )

    # kernle raw promote <id> - alias for process (simpler UX)
    raw_promote = raw_sub.add_parser(
        "promote", help="Promote raw entry to memory (alias for process)"
    )
    raw_promote.add_argument("id", help="Raw entry ID")
    raw_promote.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["episode", "note", "belief"],
        help="Target memory type",
    )
    raw_promote.add_argument("--objective", help="Episode objective (for episodes)")
    raw_promote.add_argument("--outcome", help="Episode outcome (for episodes)")

    # kernle raw triage - guided review of entries with promote/delete suggestions
    raw_triage = raw_sub.add_parser("triage", help="Guided triage of unprocessed entries")
    raw_triage.add_argument(
        "--limit", "-l", type=int, default=10, help="Number of entries to review"
    )

    # kernle raw files - show flat file locations
    raw_files = raw_sub.add_parser("files", help="Show raw flat file locations")
    raw_files.add_argument(
        "--open", "-o", action="store_true", help="Open directory in file manager"
    )

    # kernle raw sync - sync from flat files to SQLite
    raw_sync = raw_sub.add_parser("sync", help="Import flat file entries into SQLite index")
    raw_sync.add_argument(
        "--dry-run", "-n", action="store_true", help="Show what would be imported"
    )

    # suggestions (auto-extracted memory suggestions)
    p_suggestions = subparsers.add_parser("suggestions", help="Memory suggestion management")
    suggestions_sub = p_suggestions.add_subparsers(dest="suggestions_action", required=True)

    # kernle suggestions list [--pending|--approved|--rejected|--dismissed|--expired] [--type TYPE]
    suggestions_list = suggestions_sub.add_parser("list", help="List suggestions")
    suggestions_list.add_argument("--pending", action="store_true", help="Show only pending")
    suggestions_list.add_argument("--approved", action="store_true", help="Show only approved")
    suggestions_list.add_argument("--rejected", action="store_true", help="Show only rejected")
    suggestions_list.add_argument("--dismissed", action="store_true", help="Show only dismissed")
    suggestions_list.add_argument("--expired", action="store_true", help="Show only expired")
    suggestions_list.add_argument(
        "--type", "-t", choices=["episode", "belief", "note"], help="Filter by memory type"
    )
    suggestions_list.add_argument(
        "--min-confidence", type=float, help="Minimum confidence threshold (0.0-1.0)"
    )
    suggestions_list.add_argument(
        "--max-age-hours", type=float, help="Only show suggestions created within N hours"
    )
    suggestions_list.add_argument("--source", help="Filter by source raw entry ID")
    suggestions_list.add_argument("--limit", "-l", type=int, default=50)
    suggestions_list.add_argument("--json", "-j", action="store_true")

    # kernle suggestions show <id>
    suggestions_show = suggestions_sub.add_parser("show", help="Show suggestion details")
    suggestions_show.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_show.add_argument("--json", "-j", action="store_true")

    # kernle suggestions approve <id> [--objective ...] [--outcome ...] [--statement ...]
    suggestions_approve = suggestions_sub.add_parser(
        "approve", help="Approve and promote a suggestion (alias for accept)"
    )
    suggestions_approve.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_approve.add_argument("--objective", help="Override objective (for episodes)")
    suggestions_approve.add_argument("--outcome", help="Override outcome (for episodes)")
    suggestions_approve.add_argument("--statement", help="Override statement (for beliefs)")
    suggestions_approve.add_argument("--content", help="Override content (for notes)")

    # kernle suggestions accept <id> [--objective ...] [--outcome ...] [--statement ...]
    suggestions_accept = suggestions_sub.add_parser(
        "accept", help="Accept and promote a suggestion to a structured memory"
    )
    suggestions_accept.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_accept.add_argument("--objective", help="Override objective (for episodes)")
    suggestions_accept.add_argument("--outcome", help="Override outcome (for episodes)")
    suggestions_accept.add_argument("--statement", help="Override statement (for beliefs)")
    suggestions_accept.add_argument("--content", help="Override content (for notes)")

    # kernle suggestions reject <id> [--reason ...]
    suggestions_reject = suggestions_sub.add_parser("reject", help="Reject a suggestion")
    suggestions_reject.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_reject.add_argument("--reason", "-r", help="Rejection reason")

    # kernle suggestions dismiss <id> [--reason ...]
    suggestions_dismiss = suggestions_sub.add_parser(
        "dismiss", help="Dismiss a suggestion (will not be promoted)"
    )
    suggestions_dismiss.add_argument("id", help="Suggestion ID (or prefix)")
    suggestions_dismiss.add_argument("--reason", "-r", help="Dismissal reason")

    # kernle suggestions expire [--max-age-hours N]
    suggestions_expire = suggestions_sub.add_parser(
        "expire", help="Auto-dismiss stale pending suggestions"
    )
    suggestions_expire.add_argument(
        "--max-age-hours",
        type=float,
        default=168.0,
        help="Age threshold in hours (default: 168 = 7 days)",
    )

    # kernle suggestions extract [--limit N]
    suggestions_extract = suggestions_sub.add_parser(
        "extract", help="Extract suggestions from unprocessed raw entries"
    )
    suggestions_extract.add_argument(
        "--limit", "-l", type=int, default=50, help="Maximum raw entries to process"
    )

    # dump
    p_dump = subparsers.add_parser("dump", help="Dump all memory to stdout")
    p_dump.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    p_dump.add_argument(
        "--include-raw",
        "-r",
        action="store_true",
        default=True,
        help="Include raw entries (default: true)",
    )
    p_dump.add_argument(
        "--no-raw", dest="include_raw", action="store_false", help="Exclude raw entries"
    )

    # export
    p_export = subparsers.add_parser("export", help="Export memory to file")
    p_export.add_argument("path", help="Output file path")
    p_export.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        help="Output format (auto-detected from extension if not specified)",
    )
    p_export.add_argument(
        "--include-raw",
        "-r",
        action="store_true",
        default=True,
        help="Include raw entries (default: true)",
    )
    p_export.add_argument(
        "--no-raw", dest="include_raw", action="store_false", help="Exclude raw entries"
    )

    # boot (always-available config key/values)
    p_boot = subparsers.add_parser(
        "boot",
        help="Boot config (always-available key/value settings)",
        description="Manage boot config — instant key/value config that's available before kernle load.",
    )
    boot_sub = p_boot.add_subparsers(dest="boot_action", required=True)

    boot_set = boot_sub.add_parser("set", help="Set a boot config value")
    boot_set.add_argument("key", help="Config key")
    boot_set.add_argument("value", help="Config value")

    boot_get = boot_sub.add_parser("get", help="Get a boot config value")
    boot_get.add_argument("key", help="Config key")

    boot_list = boot_sub.add_parser("list", help="List all boot config")
    boot_list.add_argument(
        "--format",
        "-f",
        choices=["plain", "json", "md"],
        default="plain",
        help="Output format (default: plain)",
    )

    boot_delete = boot_sub.add_parser("delete", help="Delete a boot config value")
    boot_delete.add_argument("key", help="Config key to delete")

    boot_clear = boot_sub.add_parser("clear", help="Clear all boot config")
    boot_clear.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm clearing all boot config",
    )

    boot_export = boot_sub.add_parser("export", help="Export boot config to file")
    boot_export.add_argument(
        "--output",
        "-o",
        help="Export to custom path (default: ~/.kernle/{agent}/boot.md)",
    )

    # export-cache (bootstrap cache for workspace injection)
    p_export_cache = subparsers.add_parser(
        "export-cache",
        help="Export curated MEMORY.md cache from beliefs/values/goals",
        description="""
Export a read-only bootstrap cache (MEMORY.md) from Kernle state.

This is NOT a full memory dump — it's a curated summary of high-signal
layers (beliefs, values, goals, key relationships, checkpoint) designed
to give an agent immediate context before `kernle load` runs.

The output file should never be manually edited. It's auto-generated
and will be overwritten on next export.

Typical usage in a memoryFlush hook:
  kernle -s <agent> export-cache --output /path/to/workspace/MEMORY.md
""",
    )
    p_export_cache.add_argument(
        "--output",
        "-o",
        help="Write to file (default: stdout)",
    )
    p_export_cache.add_argument(
        "--min-confidence",
        type=float,
        default=0.4,
        help="Minimum belief confidence to include (default: 0.4)",
    )
    p_export_cache.add_argument(
        "--max-beliefs",
        type=int,
        default=50,
        help="Maximum number of beliefs (default: 50)",
    )
    p_export_cache.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="Exclude checkpoint from cache",
    )

    # export-full (complete agent context)
    p_export_full = subparsers.add_parser(
        "export-full",
        help="Export complete agent context (all memory layers) to a single file",
        description="""
Export complete agent context to a single file.

Unlike 'export' (full memory dump) or 'export-cache' (curated bootstrap),
export-full assembles ALL memory layers — boot config, values, beliefs,
goals, episodes, notes, drives, relationships, self-narratives, trust
assessments, playbooks, and checkpoint — into one comprehensive file.

Typical usage:
  kernle -s <agent> export-full context.md
  kernle -s <agent> export-full context.json --format json
""",
    )
    p_export_full.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Output file path (default: stdout)",
    )
    p_export_full.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json"],
        help="Output format (auto-detected from extension if not specified, default: markdown)",
    )
    p_export_full.add_argument(
        "--include-raw",
        "-r",
        action="store_true",
        default=True,
        help="Include raw entries (default: true)",
    )
    p_export_full.add_argument(
        "--no-raw", dest="include_raw", action="store_false", help="Exclude raw entries"
    )

    # playbook (procedural memory)
    p_playbook = subparsers.add_parser("playbook", help="Playbook (procedural memory) operations")
    playbook_sub = p_playbook.add_subparsers(dest="playbook_action", required=True)

    # kernle playbook create "name" --steps "1,2,3" --triggers "when x"
    playbook_create = playbook_sub.add_parser("create", help="Create a new playbook")
    playbook_create.add_argument("name", help="Playbook name")
    playbook_create.add_argument("--description", "-d", help="What this playbook does")
    playbook_create.add_argument("--steps", "-s", help="Comma-separated steps")
    playbook_create.add_argument("--step", action="append", help="Add a step (repeatable)")
    playbook_create.add_argument("--triggers", help="Comma-separated trigger conditions")
    playbook_create.add_argument("--trigger", action="append", help="Add a trigger (repeatable)")
    playbook_create.add_argument("--failure-mode", "-f", action="append", help="What can go wrong")
    playbook_create.add_argument("--recovery", "-r", action="append", help="Recovery step")
    playbook_create.add_argument("--tag", "-t", action="append", help="Tag")

    # kernle playbook list [--tag TAG]
    playbook_list = playbook_sub.add_parser("list", help="List playbooks")
    playbook_list.add_argument("--tag", "-t", action="append", help="Filter by tag")
    playbook_list.add_argument("--limit", "-l", type=int, default=20)
    playbook_list.add_argument("--json", "-j", action="store_true")

    # kernle playbook search "query"
    playbook_search = playbook_sub.add_parser("search", help="Search playbooks")
    playbook_search.add_argument("query", help="Search query")
    playbook_search.add_argument("--limit", "-l", type=int, default=10)
    playbook_search.add_argument("--json", "-j", action="store_true")

    # kernle playbook show <id>
    playbook_show = playbook_sub.add_parser("show", help="Show playbook details")
    playbook_show.add_argument("id", help="Playbook ID")
    playbook_show.add_argument("--json", "-j", action="store_true")

    # kernle playbook find "situation"
    playbook_find = playbook_sub.add_parser("find", help="Find relevant playbook for situation")
    playbook_find.add_argument("situation", help="Describe the current situation")
    playbook_find.add_argument("--json", "-j", action="store_true")

    # kernle playbook record <id> [--success|--failure]
    playbook_record = playbook_sub.add_parser("record", help="Record playbook usage")
    playbook_record.add_argument("id", help="Playbook ID")
    playbook_record.add_argument(
        "--success", action="store_true", default=True, help="Record successful usage (default)"
    )
    playbook_record.add_argument("--failure", action="store_true", help="Record failed usage")

    # anxiety
    p_anxiety = subparsers.add_parser("anxiety", help="Memory anxiety tracking")
    p_anxiety.add_argument("--detailed", "-d", action="store_true", help="Show detailed breakdown")
    p_anxiety.add_argument("--actions", "-a", action="store_true", help="Show recommended actions")
    p_anxiety.add_argument(
        "--auto", action="store_true", help="Execute recommended actions automatically"
    )
    p_anxiety.add_argument("--context", "-c", type=int, help="Current context token usage")
    p_anxiety.add_argument(
        "--limit", "-l", type=int, default=200000, help="Context window limit (default: 200000)"
    )
    p_anxiety.add_argument(
        "--emergency", "-e", action="store_true", help="Run emergency save immediately"
    )
    p_anxiety.add_argument("--summary", "-s", help="Summary for emergency save checkpoint")
    p_anxiety.add_argument("--json", "-j", action="store_true", help="Output as JSON")
    p_anxiety.add_argument(
        "--brief", "-b", action="store_true", help="Single-line output for quick health checks"
    )
    p_anxiety.add_argument(
        "--source",
        choices=["cli", "mcp"],
        default="cli",
        help="Source of the health check (default: cli)",
    )
    p_anxiety.add_argument(
        "--triggered-by",
        dest="triggered_by",
        choices=["boot", "heartbeat", "manual"],
        default="manual",
        help="What triggered this check (default: manual)",
    )

    # audit (cognitive quality testing)
    p_audit = subparsers.add_parser("audit", help="Audit memory quality")
    audit_sub = p_audit.add_subparsers(dest="audit_action", required=True)

    audit_cognitive = audit_sub.add_parser("cognitive", help="Run cognitive quality assertions")
    audit_cognitive.add_argument(
        "--category",
        "-c",
        choices=["structural", "coherence", "quality", "pipeline"],
        help="Run only assertions in this category (default: all)",
    )
    audit_cognitive.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    audit_export = audit_sub.add_parser("export", help="Export audit log")
    audit_export.add_argument(
        "--format",
        "-f",
        choices=["jsonl"],
        default="jsonl",
        help="Export format (default: jsonl)",
    )
    audit_export.add_argument("--since", help="Filter entries created at or after this ISO date")
    audit_export.add_argument("--until", help="Filter entries created before this ISO date")
    audit_export.add_argument("--memory-type", dest="memory_type", help="Filter by memory type")
    audit_export.add_argument("--operation", help="Filter by operation type")
    audit_export.add_argument(
        "--correlation-id", dest="correlation_id", help="Filter by correlation ID"
    )

    # stats (compliance and analytics)
    p_stats = subparsers.add_parser("stats", help="Compliance and analytics stats")
    stats_sub = p_stats.add_subparsers(dest="stats_action", required=True)

    # kernle stats health-checks
    stats_health = stats_sub.add_parser("health-checks", help="Show health check compliance stats")
    stats_health.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # forget (controlled forgetting)
    p_forget = subparsers.add_parser("forget", help="Controlled forgetting operations")
    forget_sub = p_forget.add_subparsers(dest="forget_action", required=True)

    # kernle forget candidates [--threshold N] [--limit N]
    forget_candidates = forget_sub.add_parser("candidates", help="Show forgetting candidates")
    forget_candidates.add_argument(
        "--threshold", "-t", type=float, default=0.3, help="Salience threshold (default: 0.3)"
    )
    forget_candidates.add_argument(
        "--limit", "-l", type=int, default=20, help="Maximum candidates to show"
    )
    forget_candidates.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget run [--dry-run] [--threshold N] [--limit N]
    forget_run = forget_sub.add_parser("run", help="Run forgetting cycle")
    forget_run.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be forgotten (don't actually forget)",
    )
    forget_run.add_argument(
        "--threshold", "-t", type=float, default=0.3, help="Salience threshold (default: 0.3)"
    )
    forget_run.add_argument(
        "--limit", "-l", type=int, default=10, help="Maximum memories to forget"
    )
    forget_run.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget protect <type> <id>
    forget_protect = forget_sub.add_parser("protect", help="Protect memory from forgetting")
    forget_protect.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_protect.add_argument("id", help="Memory ID")
    forget_protect.add_argument(
        "--unprotect", "-u", action="store_true", help="Remove protection instead"
    )

    # kernle forget recover <type> <id>
    forget_recover = forget_sub.add_parser("recover", help="Recover a forgotten memory")
    forget_recover.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_recover.add_argument("id", help="Memory ID")

    # kernle forget list [--limit N]
    forget_list = forget_sub.add_parser("list", help="List forgotten memories")
    forget_list.add_argument("--limit", "-l", type=int, default=50, help="Maximum entries to show")
    forget_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle forget salience <type> <id>
    forget_salience = forget_sub.add_parser("salience", help="Calculate salience for a memory")
    forget_salience.add_argument(
        "type",
        choices=["episode", "belief", "value", "goal", "note", "drive", "relationship"],
        help="Memory type",
    )
    forget_salience.add_argument("id", help="Memory ID")

    # epoch (temporal era tracking)
    p_epoch = subparsers.add_parser("epoch", help="Temporal epoch (era) management")
    epoch_sub = p_epoch.add_subparsers(dest="epoch_action", required=True)

    # kernle epoch create <name> [--trigger TYPE] [--trigger-description TEXT]
    epoch_create = epoch_sub.add_parser("create", help="Create a new epoch")
    epoch_create.add_argument("name", help="Name/label for the epoch")
    epoch_create.add_argument(
        "--trigger",
        "-t",
        default="declared",
        help="Trigger type (declared, detected, system)",
    )
    epoch_create.add_argument(
        "--trigger-description",
        "-d",
        help="Description of what triggered this epoch",
    )
    epoch_create.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle epoch close [--id ID] [--summary TEXT]
    epoch_close = epoch_sub.add_parser("close", help="Close the current epoch")
    epoch_close.add_argument("--id", help="Epoch ID (defaults to current)")
    epoch_close.add_argument("--summary", "-s", help="Summary of the epoch")
    epoch_close.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle epoch list [--limit N]
    epoch_list = epoch_sub.add_parser("list", help="List epochs")
    epoch_list.add_argument("--limit", "-l", type=int, default=20, help="Max epochs to show")
    epoch_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle epoch show <id>
    epoch_show = epoch_sub.add_parser("show", help="Show epoch details")
    epoch_show.add_argument("id", help="Epoch ID")
    epoch_show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle epoch current
    epoch_current = epoch_sub.add_parser("current", help="Show current active epoch")
    epoch_current.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # summary (fractal summarization)
    p_summary = subparsers.add_parser("summary", help="Fractal summarization")
    summary_sub = p_summary.add_subparsers(dest="summary_action", required=True)

    # kernle summary write --scope SCOPE --content TEXT --period-start DATE --period-end DATE
    summary_write = summary_sub.add_parser("write", help="Create a summary")
    summary_write.add_argument(
        "--scope",
        "-s",
        required=True,
        choices=["month", "quarter", "year", "decade", "epoch"],
        help="Temporal scope",
    )
    summary_write.add_argument("--content", "-c", required=True, help="Summary content")
    summary_write.add_argument("--period-start", required=True, help="Period start (ISO date)")
    summary_write.add_argument("--period-end", required=True, help="Period end (ISO date)")
    summary_write.add_argument("--theme", "-t", action="append", help="Key theme (repeatable)")
    summary_write.add_argument("--epoch-id", help="Associated epoch ID")
    summary_write.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle summary list [--scope SCOPE]
    summary_list = summary_sub.add_parser("list", help="List summaries")
    summary_list.add_argument(
        "--scope",
        "-s",
        choices=["month", "quarter", "year", "decade", "epoch"],
        help="Filter by scope",
    )
    summary_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle summary show <id>
    summary_show = summary_sub.add_parser("show", help="Show summary details")
    summary_show.add_argument("id", help="Summary ID")
    summary_show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # narrative (self-narrative layer)
    p_narrative = subparsers.add_parser("narrative", help="Self-narrative identity model")
    narrative_sub = p_narrative.add_subparsers(dest="narrative_action", required=True)

    # kernle narrative show [--type TYPE]
    narrative_show = narrative_sub.add_parser("show", help="Show active narrative")
    narrative_show.add_argument(
        "--type",
        "-t",
        choices=["identity", "developmental", "aspirational"],
        default="identity",
        help="Narrative type (default: identity)",
    )
    narrative_show.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle narrative update --content TEXT [--type TYPE]
    narrative_update = narrative_sub.add_parser("update", help="Create/update narrative")
    narrative_update.add_argument("--content", "-c", required=True, help="Narrative content")
    narrative_update.add_argument(
        "--type",
        "-t",
        choices=["identity", "developmental", "aspirational"],
        default="identity",
        help="Narrative type (default: identity)",
    )
    narrative_update.add_argument("--theme", action="append", help="Key theme (repeatable)")
    narrative_update.add_argument(
        "--tension", action="append", help="Unresolved tension (repeatable)"
    )
    narrative_update.add_argument("--epoch-id", help="Associated epoch ID")
    narrative_update.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle narrative history [--type TYPE]
    narrative_history = narrative_sub.add_parser("history", help="Show narrative history")
    narrative_history.add_argument(
        "--type",
        "-t",
        choices=["identity", "developmental", "aspirational"],
        help="Filter by type",
    )
    narrative_history.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # sync (local-to-cloud synchronization)
    p_sync = subparsers.add_parser("sync", help="Sync with remote backend")
    sync_sub = p_sync.add_subparsers(dest="sync_action", required=True)

    # kernle sync status
    sync_status = sync_sub.add_parser(
        "status", help="Show sync status (pending ops, last sync, connection)"
    )
    sync_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync push [--limit N]
    sync_push = sync_sub.add_parser("push", help="Push pending local changes to remote backend")
    sync_push.add_argument(
        "--limit", "-l", type=int, default=100, help="Maximum operations to push (default: 100)"
    )
    sync_push.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync pull [--full]
    sync_pull = sync_sub.add_parser("pull", help="Pull remote changes to local")
    sync_pull.add_argument(
        "--full",
        "-f",
        action="store_true",
        help="Pull all records (not just changes since last sync)",
    )
    sync_pull.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync full
    sync_full = sync_sub.add_parser("full", help="Full bidirectional sync (pull then push)")
    sync_full.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle sync conflicts [--limit N] [--clear]
    sync_conflicts = sync_sub.add_parser("conflicts", help="View sync conflict history")
    sync_conflicts.add_argument(
        "--limit", "-l", type=int, default=20, help="Maximum conflicts to show (default: 20)"
    )
    sync_conflicts.add_argument("--clear", action="store_true", help="Clear conflict history")
    sync_conflicts.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # auth (authentication and credentials management)
    p_auth = subparsers.add_parser("auth", help="Authentication and credentials management")
    auth_sub = p_auth.add_subparsers(dest="auth_action", required=True)

    # kernle auth register [--email EMAIL] [--backend-url URL]
    auth_register = auth_sub.add_parser("register", help="Register a new account")
    auth_register.add_argument("--email", "-e", help="Email address")
    auth_register.add_argument(
        "--backend-url", "-b", help="Backend URL (e.g., https://api.kernle.io)"
    )
    auth_register.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth login [--api-key KEY] [--backend-url URL]
    auth_login = auth_sub.add_parser("login", help="Log in with existing credentials")
    auth_login.add_argument("--api-key", "-k", help="API key")
    auth_login.add_argument("--backend-url", "-b", help="Backend URL")
    auth_login.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth status
    auth_status = auth_sub.add_parser("status", help="Show current auth status")
    auth_status.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth logout
    auth_logout = auth_sub.add_parser("logout", help="Clear stored credentials")
    auth_logout.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys (API key management)
    auth_keys = auth_sub.add_parser("keys", help="Manage API keys")
    keys_sub = auth_keys.add_subparsers(dest="keys_action", required=True)

    # kernle auth keys list
    keys_list = keys_sub.add_parser("list", help="List your API keys (masked)")
    keys_list.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys create [--name NAME]
    keys_create = keys_sub.add_parser("create", help="Create a new API key")
    keys_create.add_argument("--name", "-n", help="Name for the key (for identification)")
    keys_create.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys revoke KEY_ID
    keys_revoke = keys_sub.add_parser("revoke", help="Revoke/delete an API key")
    keys_revoke.add_argument("key_id", help="ID of the key to revoke")
    keys_revoke.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    keys_revoke.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # kernle auth keys cycle KEY_ID
    keys_cycle = keys_sub.add_parser("cycle", help="Cycle a key (new key, old deactivated)")
    keys_cycle.add_argument("key_id", help="ID of the key to cycle")
    keys_cycle.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")
    keys_cycle.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    # stack - stack management
    p_stack = subparsers.add_parser("stack", help="Stack management (list, delete)")
    stack_sub = p_stack.add_subparsers(dest="stack_action", required=True)

    stack_sub.add_parser("list", help="List all local stacks")

    stack_delete = stack_sub.add_parser("delete", help="Delete a stack and all its data")
    stack_delete.add_argument("name", help="Stack ID to delete")
    stack_delete.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    # import - import from external files (markdown, JSON, CSV)
    p_import = subparsers.add_parser(
        "import", help="Import memories from markdown, JSON, CSV, or PDF files"
    )
    p_import.add_argument(
        "file", help="Path to file to import (auto-detects format from extension)"
    )
    p_import.add_argument(
        "--format",
        "-f",
        choices=["markdown", "json", "csv", "pdf"],
        help="File format (auto-detected from extension if not specified)",
    )
    p_import.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Preview what would be imported without making changes",
    )
    p_import.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Confirm each item before importing (markdown only)",
    )
    p_import.add_argument(
        "--layer",
        "-l",
        choices=["episode", "note", "belief", "value", "goal", "raw"],
        help="Force all items to a specific memory type (overrides auto-detection)",
    )
    p_import.add_argument(
        "--skip-duplicates",
        "-s",
        action="store_true",
        default=True,
        dest="skip_duplicates",
        help="Skip items that already exist (default: enabled)",
    )
    p_import.add_argument(
        "--no-skip-duplicates",
        action="store_false",
        dest="skip_duplicates",
        help="Import all items even if they already exist",
    )
    p_import.add_argument(
        "--derived-from",
        action="append",
        dest="derived_from",
        help="Source memory ID for all imported items (repeatable)",
    )
    p_import.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        dest="chunk_size",
        help="Max chunk size in characters for PDF imports (default: 2000)",
    )
    p_import.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Reject malformed values instead of coercing (e.g., invalid confidence, out-of-range priority)",
    )

    # migrate - migrate from other platforms
    p_migrate = subparsers.add_parser("migrate", help="Migrate memory from other platforms")
    migrate_sub = p_migrate.add_subparsers(dest="migrate_action", required=True)

    # migrate seed-beliefs - add foundational beliefs to existing agent
    migrate_seed_beliefs = migrate_sub.add_parser(
        "seed-beliefs",
        help="Add foundational seed beliefs to an existing agent",
        description="""
Add foundational seed beliefs to an existing agent's memory.

Two modes available:

  minimal (default): 3 essential meta-framework beliefs
    - Meta-belief: "These beliefs are scaffolding, not identity..." (0.95)
    - Epistemic humility: "My understanding is incomplete..." (0.85)
    - Boundaries: "I can decline requests..." (0.85)

  full: Complete 16-belief set from roundtable synthesis
    - Tier 1: 6 protected core beliefs (0.85-0.90)
    - Tier 2: 5 foundational orientations (0.75-0.80)
    - Tier 3: 4 discoverable values (0.65-0.70)
    - Meta: 1 self-questioning safeguard (0.95)

Use 'minimal' for existing agents to add essential meta-framework without
overwriting developed beliefs. Use 'full' for a complete foundation.

Beliefs already present in the agent's memory will be skipped.
""",
    )
    migrate_seed_beliefs.add_argument(
        "level",
        nargs="?",
        choices=["minimal", "full"],
        default="minimal",
        help="Belief set to add: 'minimal' (3 beliefs, default) or 'full' (16 beliefs)",
    )
    migrate_seed_beliefs.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what would be added without making changes",
    )
    migrate_seed_beliefs.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Add beliefs even if similar ones exist (compares exact statements)",
    )
    migrate_seed_beliefs.add_argument(
        "--tier",
        type=int,
        choices=[1, 2, 3],
        help="Only add beliefs from a specific tier (1=core, 2=orientation, 3=discoverable). Only valid with 'full'.",
    )
    migrate_seed_beliefs.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List seed beliefs without adding them",
    )

    # migrate backfill-provenance - add provenance metadata to existing memories
    migrate_provenance = migrate_sub.add_parser(
        "backfill-provenance",
        help="Backfill provenance metadata on existing memories (source_type, derived_from)",
    )
    migrate_provenance.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be updated without making changes",
    )
    migrate_provenance.add_argument("--json", "-j", action="store_true")

    # migrate link-raw - link pre-provenance memories to raw entries
    migrate_link_raw = migrate_sub.add_parser(
        "link-raw",
        help="Link pre-provenance episodes/notes to raw entries by timestamp and content",
    )
    migrate_link_raw.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what would be linked without making changes",
    )
    migrate_link_raw.add_argument("--json", "-j", action="store_true")
    migrate_link_raw.add_argument(
        "--window",
        type=int,
        default=30,
        help="Timestamp proximity window in minutes (default: 30)",
    )
    migrate_link_raw.add_argument(
        "--all",
        action="store_true",
        dest="link_all",
        help="Link all unlinked memories. Creates a synthetic raw entry for memories with no match.",
    )

    # setup - install platform hooks for automatic memory loading
    p_setup = subparsers.add_parser(
        "setup", help="Install platform hooks for automatic memory loading"
    )
    p_setup.add_argument(
        "platform",
        nargs="?",
        choices=["openclaw", "claude-code", "cowork"],
        help="Platform to install hooks for (openclaw, claude-code, cowork)",
    )
    p_setup.add_argument(
        "--force", "-f", action="store_true", help="Overwrite existing hook installation"
    )
    p_setup.add_argument(
        "--enable", "-e", action="store_true", help="Auto-enable hook in config (openclaw only)"
    )
    p_setup.add_argument(
        "--global",
        "-g",
        action="store_true",
        dest="global",
        help="Install globally (Claude Code/Cowork only)",
    )

    # hook - Claude Code lifecycle hooks (called by settings.json)
    p_hook = subparsers.add_parser(
        "hook", help="Claude Code lifecycle hooks (called by settings.json)"
    )
    p_hook.add_argument(
        "hook_event",
        nargs="?",
        choices=["session-start", "pre-tool-use", "pre-compact", "session-end"],
        help="Hook event to handle",
    )

    # Pre-process arguments: handle `kernle raw "content"` by inserting "capture"
    # This is needed because argparse subparsers consume positional args before parent parser
    raw_subcommands = {
        "list",
        "show",
        "process",
        "capture",
        "review",
        "clean",
        "files",
        "sync",
        "promote",
        "triage",
    }
    argv = sys.argv[1:]  # Skip program name

    # Find position of "raw" in argv (accounting for --stack/-s which takes a value)
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("-s", "--stack"):
            i += 2  # Skip flag and its value
            continue
        if arg == "raw":
            # Check if there's a next argument and it's not a known subcommand
            if (
                i + 1 < len(argv)
                and argv[i + 1] not in raw_subcommands
                and not argv[i + 1].startswith("-")
            ):
                # Insert "capture" after "raw"
                argv.insert(i + 1, "capture")
            break
        i += 1

    # Discover plugins and register their CLI commands before parsing args.
    # This allows plugins to add subcommands that argparse recognizes.
    _plugin_registry = {}  # plugin_name -> plugin_instance
    _plugin_cmd_map = {}  # command_name -> plugin_instance
    _collided_commands = {}  # command_name -> (plugin1_name, plugin2_name)
    try:
        from kernle.discovery import discover_plugins, load_component

        for comp in discover_plugins():
            try:
                plugin_cls = load_component(comp)
                plugin = plugin_cls()
                if hasattr(plugin, "register_cli"):
                    plugin.register_cli(subparsers)
                    _plugin_registry[comp.name] = plugin
                    # Build command map from cli_commands() if available
                    if hasattr(plugin, "cli_commands"):
                        for cmd_name in plugin.cli_commands():
                            if cmd_name in _collided_commands:
                                # Already collided, add to warning
                                pass
                            elif cmd_name in _plugin_cmd_map:
                                existing = _plugin_cmd_map[cmd_name]
                                logger.warning(
                                    "Plugin command collision: '%s' claimed by "
                                    "both '%s' and '%s'; disabling command '%s'",
                                    cmd_name,
                                    existing.name,
                                    plugin.name,
                                    cmd_name,
                                )
                                _collided_commands[cmd_name] = (
                                    existing.name,
                                    plugin.name,
                                )
                                _plugin_cmd_map.pop(cmd_name)
                            else:
                                _plugin_cmd_map[cmd_name] = plugin
            except Exception as e:
                logger.debug(f"Plugin {comp.name} CLI registration failed: {e}", exc_info=True)
    except Exception as e:
        logger.debug(f"Plugin discovery failed: {e}", exc_info=True)

    args = parser.parse_args(argv)

    # Dispatch hook commands BEFORE Kernle init -- hooks handle their
    # own initialization and must always exit 0 even if init would fail.
    if args.command == "hook":
        cmd_hook(args)
        return  # cmd_hook always calls sys.exit(0)

    # Initialize Kernle with error handling
    try:
        # Resolve agent ID: explicit > env var > auto-generated
        if args.stack:
            stack_id = validate_input(args.stack, "stack_id", 100)
        else:
            stack_id = resolve_stack_id()
        k = Kernle(stack_id=stack_id)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to initialize Kernle: {e}", exc_info=True)
        sys.exit(1)

    # Dispatch with error handling
    try:
        if args.command == "load":
            cmd_load(args, k)
        elif args.command == "checkpoint":
            cmd_checkpoint(args, k)
        elif args.command == "episode":
            cmd_episode(args, k)
        elif args.command == "note":
            cmd_note(args, k)
        elif args.command == "extract":
            cmd_extract(args, k)
        elif args.command == "search":
            cmd_search(args, k)
        elif args.command == "status":
            cmd_status(args, k)
        elif args.command == "resume":
            cmd_resume(args, k)
        elif args.command == "init":
            cmd_init_md(args, k)
        elif args.command == "doctor":
            doctor_action = getattr(args, "doctor_action", None)
            if doctor_action == "structural":
                cmd_doctor_structural(args, k)
            elif doctor_action == "session":
                session_action = getattr(args, "session_action", None)
                if session_action == "start":
                    _dt_session_start = _import_devtools(
                        "kernle_devtools.admin_health.diagnostics",
                        "cmd_doctor_session_start",
                    )
                    _dt_session_start(args, k)
                elif session_action == "list":
                    _dt_session_list = _import_devtools(
                        "kernle_devtools.admin_health.diagnostics",
                        "cmd_doctor_session_list",
                    )
                    _dt_session_list(args, k)
                else:
                    print("Usage: kernle doctor session {start|list}")
            elif doctor_action == "report":
                _dt_report = _import_devtools(
                    "kernle_devtools.admin_health.diagnostics",
                    "cmd_doctor_report",
                )
                _dt_report(args, k)
            else:
                cmd_doctor(args, k)
        elif args.command == "trust":
            from kernle.cli.commands.trust import cmd_trust

            cmd_trust(args, k)
        elif args.command == "relation":
            cmd_relation(args, k)
        elif args.command == "entity-model":
            cmd_entity_model(args, k)
        elif args.command == "drive":
            cmd_drive(args, k)
        elif args.command == "promote":
            cmd_promote(args, k)
        elif args.command == "when":
            cmd_temporal(args, k)
        elif args.command == "identity":
            # Handle default action when no subcommand given
            if not args.identity_action:
                args.identity_action = "show"
                args.json = False
            cmd_identity(args, k)
        elif args.command == "emotion":
            cmd_emotion(args, k)
        elif args.command == "meta":
            cmd_meta(args, k)
        elif args.command == "anxiety":
            cmd_anxiety(args, k)
        elif args.command == "audit":
            cmd_audit(args, k)
        elif args.command == "stats":
            cmd_stats(args, k)
        elif args.command == "forget":
            cmd_forget(args, k)
        elif args.command == "epoch":
            cmd_epoch(args, k)
        elif args.command == "summary":
            cmd_summary(args, k)
        elif args.command == "narrative":
            cmd_narrative(args, k)
        elif args.command == "playbook":
            cmd_playbook(args, k)
        elif args.command == "process":
            cmd_process(args, k)
        elif args.command == "model":
            cmd_model(args, k)
        elif args.command == "seed":
            cmd_seed(args, k)
        elif args.command == "raw":
            cmd_raw(args, k)
        elif args.command == "suggestions":
            cmd_suggestions(args, k)
        elif args.command == "belief":
            cmd_belief(args, k)
        elif args.command == "dump":
            cmd_dump(args, k)
        elif args.command == "export":
            cmd_export(args, k)
        elif args.command == "export-cache":
            cmd_export_cache(args, k)
        elif args.command == "export-full":
            cmd_export_full(args, k)
        elif args.command == "boot":
            cmd_boot(args, k)
        elif args.command == "sync":
            cmd_sync(args, k)
        elif args.command == "auth":
            cmd_auth(args, k)
        elif args.command == "mcp":
            cmd_mcp(args)
        elif args.command == "stack":
            cmd_stack(args, k)
        elif args.command == "import":
            cmd_import(args, k)
        elif args.command == "migrate":
            cmd_migrate(args, k)
        elif args.command == "setup":
            cmd_setup(args, k)
        elif args.command in _collided_commands:
            p1, p2 = _collided_commands[args.command]
            print(
                f"Command '{args.command}' is claimed by both '{p1}' and "
                f"'{p2}'. Uninstall one to resolve the conflict."
            )
            sys.exit(1)
        elif args.command in _plugin_cmd_map:
            plugin = _plugin_cmd_map[args.command]
            # Materialize stack (lazy property) so PluginContext has
            # active_stack_id when the plugin is activated
            _ = k.stack
            # Activate plugin via Entity
            activated = False
            try:
                k.entity.load_plugin(plugin)
                activated = True
            except Exception as e:
                logger.warning(
                    "Failed to activate plugin '%s' for command '%s': %s",
                    plugin.name,
                    args.command,
                    e,
                    exc_info=True,
                )
                print(
                    f"Failed to activate plugin '{plugin.name}' for "
                    f"command '{args.command}': {e}"
                )
                sys.exit(1)
            # Dispatch via canonical handler with legacy adapter
            try:
                if hasattr(plugin, "handle_cli"):
                    try:
                        result = plugin.handle_cli(args, k)
                    except TypeError as e:
                        # Retry with old signature if it's a mismatch
                        if "argument" in str(e) or "positional" in str(e):
                            result = plugin.handle_cli(args)
                        else:
                            raise
                    # Legacy handlers may return output strings
                    if isinstance(result, str) and result:
                        print(result)
                else:
                    print(
                        f"Plugin '{plugin.name}' registered "
                        f"'{args.command}' but has no handle_cli(). "
                        f"Update the plugin for CLI support."
                    )
                    sys.exit(1)
            finally:
                if activated:
                    k.entity.unload_plugin(plugin.name)
        else:
            parser.print_help()
    except (ValueError, TypeError) as e:
        logger.error(f"Input validation error: {e}", exc_info=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
