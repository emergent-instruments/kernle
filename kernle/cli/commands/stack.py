"""Stack management commands (list, delete)."""

import logging
import shutil
from typing import TYPE_CHECKING

from kernle.storage.sqlite import SQLiteStorage
from kernle.utils import get_kernle_home

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse

    from kernle import Kernle


def cmd_stack(args: "argparse.Namespace", k: "Kernle") -> None:
    """Handle agent subcommands."""
    action = args.stack_action

    if action == "list":
        _list_stacks(args, k)
    elif action == "delete":
        _delete_stack(args, k)


def _list_stacks(args: "argparse.Namespace", k: "Kernle") -> None:
    """List all local agents."""
    kernle_dir = get_kernle_home()

    if not kernle_dir.exists():
        print("No agents found (Kernle not initialized)")
        return

    # Find agent directories (those with memory.db or raw/ subdirectory)
    agents = []

    # Check for multi-agent SQLite structure via storage layer
    if isinstance(k._storage, SQLiteStorage):
        try:
            agents = list(k._storage.list_stack_ids())
        except Exception as e:
            logger.debug(f"Failed to query agents from database: {e}", exc_info=True)

    # Also check for per-agent directories (for raw layer)
    for item in kernle_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            # Skip non-agent directories
            if item.name in ("logs", "cache", "__pycache__"):
                continue
            if item.name not in agents:
                agents.append(item.name)

    if not agents:
        print("No agents found")
        return

    agents.sort()

    print(f"Local Stacks ({len(agents)} total)")
    print("=" * 50)

    for stack_id in agents:
        agent_dir = kernle_dir / stack_id
        raw_count = 0
        has_dir = agent_dir.exists()

        if has_dir:
            raw_dir = agent_dir / "raw"
            if raw_dir.exists():
                raw_count = sum(1 for f in raw_dir.glob("*.md"))

        # Get episode/note counts from storage layer
        episode_count = note_count = belief_count = 0
        if isinstance(k._storage, SQLiteStorage):
            try:
                counts = k._storage.get_stack_counts(stack_id)
                episode_count = counts.get("episodes", 0)
                note_count = counts.get("notes", 0)
                belief_count = counts.get("beliefs", 0)
            except Exception as e:
                logger.debug(f"Failed to get counts for agent '{stack_id}': {e}", exc_info=True)

        # Mark current agent
        marker = " ← current" if stack_id == k.stack_id else ""
        print(f"\n  {stack_id}{marker}")
        print(
            f"    Episodes: {episode_count}  Notes: {note_count}  Beliefs: {belief_count}  Raw: {raw_count}"
        )


def _delete_stack(args: "argparse.Namespace", k: "Kernle") -> None:
    """Delete an agent and all its data."""
    try:
        stack_id = k._validate_stack_id(args.name)
    except ValueError as e:
        print(f"Invalid stack name: {e}")
        return
    force = getattr(args, "force", False)

    if stack_id == k.stack_id:
        print(f"❌ Cannot delete current agent '{stack_id}'")
        print("   Switch to a different stack first with: kernle -s <other> ...")
        return

    if not isinstance(k._storage, SQLiteStorage):
        print("Stack management requires SQLite storage")
        return

    storage = k._storage
    kernle_dir = get_kernle_home()
    agent_dir = kernle_dir / stack_id

    # Check if agent exists
    has_db_data = False
    has_dir = agent_dir.exists()
    counts: dict[str, int] = {}

    try:
        counts = storage.get_stack_counts(stack_id)
        has_db_data = sum(counts.values()) > 0
    except Exception as e:
        logger.debug(f"Failed to check agent in database: {e}", exc_info=True)

    if not has_db_data and not has_dir:
        print(f"❌ Stack '{stack_id}' not found")
        return

    # Get counts for confirmation
    episode_count = counts.get("episodes", 0)
    note_count = counts.get("notes", 0)
    belief_count = counts.get("beliefs", 0)
    goal_count = counts.get("goals", 0)
    value_count = counts.get("values", 0)

    total_records = episode_count + note_count + belief_count + goal_count + value_count

    if not force:
        print(f"⚠️  About to delete agent '{stack_id}':")
        print(f"   Episodes: {episode_count}")
        print(f"   Notes: {note_count}")
        print(f"   Beliefs: {belief_count}")
        print(f"   Goals: {goal_count}")
        print(f"   Values: {value_count}")
        if has_dir:
            print(f"   Directory: {agent_dir}")
        print()
        confirm = input("Type the agent name to confirm deletion: ")
        if confirm != stack_id:
            print("❌ Deletion cancelled")
            return

    # Delete from SQLite via storage layer
    deleted_tables = []
    try:
        deleted = storage.delete_stack_data(stack_id)
        deleted_tables = [f"{table}: {count}" for table, count in deleted.items()]
    except Exception as e:
        print(f"⚠️  Error cleaning database: {e}")

    # Delete agent directory
    if has_dir:
        try:
            shutil.rmtree(agent_dir)
            print(f"✓ Deleted directory: {agent_dir}")
        except Exception as e:
            print(f"⚠️  Error deleting directory: {e}")

    print(f"✓ Stack '{stack_id}' deleted ({total_records} records)")
    if deleted_tables:
        print(f"   Cleaned: {', '.join(deleted_tables)}")
