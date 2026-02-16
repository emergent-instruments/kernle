"""Diagnostic commands for Kernle CLI — status, resume, temporal, dump, export, boot, drive."""

import logging
import sys
from typing import TYPE_CHECKING

from kernle.utils import get_kernle_home

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


def cmd_status(args, k: "Kernle"):
    """Show memory status."""
    status = k.status()
    print(f"Memory Status for {status['stack_id']}")
    print("=" * 40)
    print(f"Values:     {status['values']}")
    print(f"Beliefs:    {status['beliefs']}")
    print(f"Goals:      {status['goals']} active")
    print(f"Episodes:   {status['episodes']}")
    if "raw" in status:
        print(f"Raw:        {status['raw']}")
    print(f"Checkpoint: {'Yes' if status['checkpoint'] else 'No'}")

    # Composition info (v0.4.0 architecture)
    try:
        entity = k.entity
        print()
        print("Composition (v0.4.0)")
        print("-" * 40)
        print(f"Core ID:    {entity.core_id}")

        stacks_info = entity.stacks
        if stacks_info:
            for stack_id, info in stacks_info.items():
                active_marker = " (active)" if info.is_active else ""
                alias_label = f" [{info.alias}]" if info.alias else ""
                print(
                    f"Stack:      {stack_id}{alias_label}{active_marker} (schema v{info.schema_version})"
                )
        else:
            # Show stack from compat layer if not yet attached
            stack = k.stack
            if stack is not None:
                print(f"Stack:      {stack.stack_id} (detached, schema v{stack.schema_version})")
            else:
                print("Stack:      (none)")

        plugins = entity.discover_plugins()
        loaded_plugins = [p for p in plugins if p.is_loaded]
        if loaded_plugins:
            for p in loaded_plugins:
                print(f"Plugin:     {p.name} v{p.version}")
        elif plugins:
            print(f"Plugins:    {len(plugins)} discovered, 0 loaded")
        else:
            print("Plugins:    (none)")

        model = entity.model
        if model is None:
            from kernle.cli.commands.model import load_persisted_model

            model = load_persisted_model(k)
            if model is not None:
                entity.set_model(model)
        if model is not None:
            print(f"Model:      {model.capabilities.provider} / {model.model_id}")
        else:
            print("Model:      (none)")
    except Exception as e:
        logger.debug(f"Failed to show composition info: {e}", exc_info=True)


def cmd_resume(args, k: "Kernle"):
    """Quick 'where was I?' view - shows last task, next step, time since checkpoint."""
    from datetime import datetime, timezone

    cp = k.load_checkpoint()

    if not cp:
        print('No checkpoint found. Start fresh or run: kernle checkpoint save "your task"')
        return

    # Calculate time since checkpoint
    age_str = ""
    try:
        ts = cp.get("timestamp", "")
        if ts:
            cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = now - cp_time
            if age.days > 0:
                age_str = f"{age.days}d ago"
            elif age.seconds > 3600:
                age_str = f"{age.seconds // 3600}h ago"
            elif age.seconds > 60:
                age_str = f"{age.seconds // 60}m ago"
            else:
                age_str = "just now"
    except Exception as e:
        logger.debug(f"Failed to calculate checkpoint age: {e}", exc_info=True)
        age_str = "unknown"

    # Parse context for structured fields
    context = cp.get("context", "")
    progress = None
    next_step = None
    blocker = None

    if context:
        for part in context.split(" | "):
            if part.startswith("Progress:"):
                progress = part[9:].strip()
            elif part.startswith("Next:"):
                next_step = part[5:].strip()
            elif part.startswith("Blocker:"):
                blocker = part[8:].strip()

    # Check anxiety level
    try:
        anxiety = k.anxiety()
        anxiety_score = anxiety.get("overall_score", 0)
        if anxiety_score > 60:
            anxiety_indicator = " 🔴"
        elif anxiety_score > 30:
            anxiety_indicator = " 🟡"
        else:
            anxiety_indicator = ""
    except Exception as e:
        logger.debug(f"Failed to get anxiety score: {e}", exc_info=True)
        anxiety_indicator = ""

    # Display
    print(f"📍 Resume Point ({age_str}){anxiety_indicator}")
    print("=" * 40)
    print(f"Task: {cp.get('current_task', 'unknown')}")

    if progress:
        print(f"Progress: {progress}")

    if next_step:
        print(f"\n→ Next: {next_step}")

    if blocker:
        print(f"\n⚠ Blocker: {blocker}")

    if cp.get("pending"):
        print(f"\nPending ({len(cp['pending'])} items):")
        for p in cp["pending"][:3]:
            print(f"  • {p}")
        if len(cp["pending"]) > 3:
            print(f"  ... and {len(cp['pending']) - 3} more")

    # Stale warning
    try:
        if ts:
            cp_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age = now - cp_time
            if age.total_seconds() > 6 * 3600:
                print(f"\n⚠ Checkpoint is stale ({age_str}). Consider saving a fresh one.")
    except Exception as e:
        logger.debug(f"Failed to check checkpoint staleness: {e}", exc_info=True)


def cmd_temporal(args, k: "Kernle"):
    """Query memories by time."""
    result = k.what_happened(args.when)

    print(f"What happened {args.when}:")
    print(f"  Time range: {result['range']['start'][:10]} to {result['range']['end'][:10]}")
    print()

    if result.get("episodes"):
        print("Episodes:")
        for ep in result["episodes"][:5]:
            print(f"  - {ep['objective'][:60]} [{ep.get('outcome_type', '?')}]")

    if result.get("notes"):
        print("Notes:")
        for n in result["notes"][:5]:
            print(f"  - {n['content'][:60]}...")


def cmd_dump(args, k: "Kernle"):
    """Dump all memory to stdout."""
    include_raw = args.include_raw
    format_type = args.format

    content = k.dump(include_raw=include_raw, format=format_type)
    print(content)


def cmd_export(args, k: "Kernle"):
    """Export memory to a file."""
    include_raw = args.include_raw
    format_type = args.format

    # Auto-detect format from extension if not specified
    if not format_type:
        if args.path.endswith(".json"):
            format_type = "json"
        else:
            format_type = "markdown"

    k.export(args.path, include_raw=include_raw, format=format_type)
    print(f"✓ Exported memory to {args.path}")


def cmd_boot(args, k: "Kernle"):
    """Handle boot config subcommands."""
    action = getattr(args, "boot_action", None)

    if action == "set":
        key = args.key
        value = args.value
        k.boot_set(key, value)
        print(f"✓ {key}: {value}")

    elif action == "get":
        key = args.key
        value = k.boot_get(key)
        if value is not None:
            print(value)
        else:
            print(f"Key not found: {key}", file=sys.stderr)
            sys.exit(1)

    elif action == "list":
        config = k.boot_list()
        if not config:
            print("(no boot config)")
            return

        fmt = getattr(args, "format", "plain")
        if fmt == "json":
            import json

            print(json.dumps(config, indent=2))
        elif fmt == "md":
            print("## Boot Config")
            for key, value in sorted(config.items()):
                print(f"- {key}: {value}")
        else:
            for key, value in sorted(config.items()):
                print(f"{key}: {value}")

    elif action == "delete":
        key = args.key
        deleted = k.boot_delete(key)
        if deleted:
            print(f"✓ Deleted: {key}")
        else:
            print(f"Key not found: {key}", file=sys.stderr)
            sys.exit(1)

    elif action == "clear":
        confirm = getattr(args, "confirm", False)
        if not confirm:
            print("Use --confirm to clear all boot config", file=sys.stderr)
            sys.exit(2)
        count = k.boot_clear()
        print(f"✓ Cleared {count} boot config entries")

    elif action == "export":
        output = getattr(args, "output", None)
        k._export_boot_file()
        boot_path = get_kernle_home() / k.stack_id / "boot.md"
        if output:
            # Copy to custom location
            config = k.boot_list()
            if config:
                import shutil

                shutil.copy2(boot_path, output)
                print(f"✓ Boot config exported to {output}")
            else:
                print("(no boot config to export)")
        else:
            print(f"✓ Boot config exported to {boot_path}")


def cmd_export_cache(args, k: "Kernle"):
    """Export curated MEMORY.md bootstrap cache from Kernle state."""
    output_path = getattr(args, "output", None)
    min_confidence = getattr(args, "min_confidence", 0.4)
    max_beliefs = getattr(args, "max_beliefs", 50)
    no_checkpoint = getattr(args, "no_checkpoint", False)

    content = k.export_cache(
        path=output_path,
        min_confidence=min_confidence,
        max_beliefs=max_beliefs,
        include_checkpoint=not no_checkpoint,
    )

    if output_path:
        # Count what was exported
        belief_count = content.count("\n- [")
        value_count = content.count("\n- **")
        print(f"✓ Cache exported to {output_path}")
        print(f"  Beliefs: ~{belief_count}, Values+Relationships: ~{value_count}")
        print(f"  Min confidence: {min_confidence:.0%}")
    else:
        # Print to stdout
        print(content)


def cmd_export_full(args, k: "Kernle"):
    """Export complete agent context to a single file."""
    path = getattr(args, "path", None)
    format_type = getattr(args, "format", None)
    include_raw = getattr(args, "include_raw", True)

    # Auto-detect format from extension if not specified
    if not format_type:
        if path and path.endswith(".json"):
            format_type = "json"
        else:
            format_type = "markdown"

    content = k.export_full(path=path, format=format_type, include_raw=include_raw)

    if path:
        print(f"Exported full agent context to {path}")
    else:
        print(content)


def cmd_drive(args, k: "Kernle"):
    """Set or view drives."""
    if args.drive_action == "list":
        drives = k.load_drives()
        if not drives:
            print("No drives set.")
            return
        print("Drives:")
        for d in drives:
            focus = f" → {', '.join(d.get('focus_areas', []))}" if d.get("focus_areas") else ""
            print(f"  {d['drive_type']}: {d['intensity']:.0%}{focus}")

    elif args.drive_action == "set":
        k.drive(args.type, args.intensity, args.focus)
        print(f"✓ Drive '{args.type}' set to {args.intensity:.0%}")

    elif args.drive_action == "satisfy":
        if k.satisfy_drive(args.type, args.amount):
            print(f"✓ Satisfied drive '{args.type}'")
        else:
            print(f"Drive '{args.type}' not found")
