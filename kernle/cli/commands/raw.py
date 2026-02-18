"""Raw entry commands for Kernle CLI."""

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def resolve_raw_id(k: "Kernle", partial_id: str) -> str:
    """Resolve a partial raw entry ID to full ID.

    Tries exact match first, then database-level prefix match.
    Returns full ID or raises ValueError if not found or ambiguous.
    """
    # First try exact match
    entry = k.get_raw(partial_id)
    if entry:
        return partial_id

    # Try prefix match via database LIKE query (scalable for large stacks)
    if hasattr(k._storage, "find_raw_by_prefix"):
        matches = k._storage.find_raw_by_prefix(partial_id, limit=6)
    else:
        # Fallback for non-SQLite backends
        entries = k.list_raw(limit=1000)
        matches = [e for e in entries if e["id"].startswith(partial_id)]

    if len(matches) == 0:
        raise ValueError(f"Raw entry '{partial_id}' not found")
    elif len(matches) == 1:
        return matches[0].id if hasattr(matches[0], "id") else matches[0]["id"]
    else:
        # Multiple matches — show up to 5
        match_ids = [(m.id[:12] if hasattr(m, "id") else m["id"][:12]) for m in matches[:5]]
        total = len(matches)
        suffix = "..." if total > 5 else ""
        raise ValueError(
            f"Ambiguous ID '{partial_id}' matches {total}+ entries: {', '.join(match_ids)}{suffix}"
        )


def cmd_raw(args, k: "Kernle"):
    """Handle raw entry subcommands."""
    if args.raw_action == "capture" or args.raw_action is None:
        # Default action: capture a raw entry
        quiet = getattr(args, "quiet", False)
        stdin_mode = getattr(args, "stdin", False)

        # Get content from stdin or argument
        if stdin_mode:
            content = sys.stdin.read()
            if not content.strip():
                if not quiet:
                    print("✗ No content received from stdin")
                return
        else:
            content = args.content
            if not content:
                if not quiet:
                    print("✗ Content is required (use --stdin to read from stdin)")
                return

        # Note: blob has no length limit - let the storage layer handle size warnings
        blob = content  # Use the content as the blob
        tags = [validate_input(t, "tag", 100) for t in (args.tags.split(",") if args.tags else [])]
        tags = [t.strip() for t in tags if t.strip()]
        source = getattr(args, "source", None) or "cli"

        # Fold tags into blob text (tags parameter was removed from save_raw)
        if tags:
            blob = f"{blob}\n\n[Tags: {', '.join(tags)}]"

        raw_id = k.raw(blob=blob, source=source)

        # Quiet mode: minimal output for hook usage
        if quiet:
            print(raw_id[:8])
        else:
            print(f"✓ Raw entry captured: {raw_id[:8]}...")
            if tags:
                print(f"  Tags: {', '.join(tags)}")
            if source and source != "cli":
                print(f"  Source: {source}")

    elif args.raw_action == "list":
        # Filter by processed state
        processed = None
        if args.unprocessed:
            processed = False
        elif args.processed:
            processed = True

        entries = k.list_raw(processed=processed, limit=args.limit)

        if not entries:
            print("No raw entries found.")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
        else:
            unprocessed_count = sum(1 for e in entries if not e["processed"])
            print(f"Raw Entries ({len(entries)} total, {unprocessed_count} unprocessed)")
            print("=" * 50)
            for e in entries:
                status = "✓" if e["processed"] else "○"
                # Use captured_at (new) with timestamp (legacy) as fallback
                captured = e.get("captured_at") or e.get("timestamp") or ""
                captured_str = captured[:16] if captured else "unknown"
                # Use blob (new) with content (legacy) as fallback
                blob = e.get("blob") or e.get("content") or ""
                blob_preview = blob[:60].replace("\n", " ")
                if len(blob) > 60:
                    blob_preview += "..."
                print(f"\n{status} [{e['id'][:8]}] {captured_str}")
                print(f"  {blob_preview}")
                if e.get("tags"):
                    print(f"  Tags: {', '.join(e['tags'])}")
                if e["processed"] and e["processed_into"]:
                    print(f"  → {', '.join(e['processed_into'])}")

    elif args.raw_action == "show":
        try:
            full_id = resolve_raw_id(k, args.id)
        except ValueError as e:
            print(f"✗ {e}")
            return

        entry = k.get_raw(full_id)
        if not entry:
            print(f"Raw entry {args.id} not found.")
            return

        if args.json:
            print(json.dumps(entry, indent=2, default=str))
        else:
            status = "✓ Processed" if entry["processed"] else "○ Unprocessed"
            captured = entry.get("captured_at") or entry.get("timestamp") or "unknown"
            blob = entry.get("blob") or entry.get("content") or ""
            print(f"Raw Entry: {entry['id']}")
            print(f"Status: {status}")
            print(f"Captured: {captured}")
            print(f"Source: {entry['source']}")
            if entry.get("tags"):
                print(f"Tags: {', '.join(entry['tags'])} (deprecated)")
            print()
            print("Content:")
            print("-" * 40)
            print(blob)
            print("-" * 40)
            if entry["processed_into"]:
                print(f"\nProcessed into: {', '.join(entry['processed_into'])}")

    elif args.raw_action == "process":
        # Support batch processing with comma-separated IDs
        raw_ids = [id.strip() for id in args.id.split(",") if id.strip()]

        success_count = 0
        for raw_id in raw_ids:
            try:
                full_id = resolve_raw_id(k, raw_id)
                memory_id = k.process_raw(
                    raw_id=full_id,
                    as_type=args.type,
                    objective=args.objective,
                    outcome=args.outcome,
                )
                print(f"✓ Processed {full_id[:8]}... → {args.type}:{memory_id[:8]}...")
                success_count += 1
            except ValueError as e:
                print(f"✗ {raw_id}: {e}")

        if len(raw_ids) > 1:
            print(f"\nProcessed {success_count}/{len(raw_ids)} entries")

    elif args.raw_action == "review":
        # Guided review of unprocessed entries
        entries = k.list_raw(processed=False, limit=args.limit)

        if not entries:
            print("✓ No unprocessed raw entries - memory is up to date!")
            return

        if args.json:
            print(json.dumps(entries, indent=2, default=str))
            return

        print("## Raw Entry Review")
        print(f"Found {len(entries)} unprocessed entries to review.\n")
        print("For each entry, consider:")
        print("  - **Episode**: Significant experience with a lesson learned")
        print("  - **Note**: Important observation, decision, or fact")
        print("  - **Skip**: Keep as raw (not everything needs promotion)")
        print()
        print("=" * 60)

        for i, e in enumerate(entries, 1):
            captured = e.get("captured_at") or e.get("timestamp") or ""
            captured_str = captured[:16] if captured else "unknown"
            blob = e.get("blob") or e.get("content") or ""
            print(f"\n[{i}/{len(entries)}] {captured_str} - ID: {e['id'][:8]}")
            print("-" * 40)
            print(blob)
            print("-" * 40)
            if e.get("tags"):
                print(f"Tags: {', '.join(e['tags'])} (deprecated)")

            # Provide promotion suggestions based on blob content
            content_lower = blob.lower()
            suggestions = []
            if any(
                word in content_lower for word in ["learned", "lesson", "realized", "discovered"]
            ):
                suggestions.append("episode (contains learning)")
            if any(word in content_lower for word in ["decided", "decision", "chose", "will"]):
                suggestions.append("note (contains decision)")

            if suggestions:
                print(f"💡 Suggestions: {', '.join(suggestions)}")

            print(
                f"\nTo promote: kernle -s {k.stack_id} raw process {e['id'][:8]} --type <episode|note>"
            )

        print("\n" + "=" * 60)
        print(f"\nReviewed {len(entries)} entries. Promote the meaningful ones, skip the rest.")

    elif args.raw_action == "clean":
        # Clean up old unprocessed raw entries
        age_days = getattr(args, "age", 7) or 7
        junk_mode = getattr(args, "junk", False)
        dry_run = not getattr(args, "confirm", False)

        entries = k.list_raw(processed=False, limit=500)
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=age_days)

        # Junk detection patterns
        junk_keywords = [
            "test",
            "testing",
            "list",
            "show me",
            "show",
            "help",
            "hi",
            "hello",
            "asdf",
            "aaa",
            "xxx",
            "foo",
            "bar",
            "baz",
            "123",
            "abc",
        ]

        def is_junk(entry):
            """Detect likely junk entries."""
            # Use blob (new) with content (legacy) as fallback
            blob = entry.get("blob") or entry.get("content", "")
            blob = blob.strip().lower()
            # Very short content
            if len(blob) < 10:
                return True
            # Exact match to junk keywords
            if blob in junk_keywords:
                return True
            # Starts with test-like patterns
            if blob.startswith(("test ", "testing ")):
                return True
            return False

        stale_entries = []
        junk_entries = []

        for entry in entries:
            # Check for junk first if junk mode
            if junk_mode and is_junk(entry):
                junk_entries.append(entry)
                continue

            # Check age for stale entries
            if not junk_mode:
                try:
                    # Use captured_at (new) with timestamp (legacy) as fallback
                    ts = entry.get("captured_at") or entry.get("timestamp", "")
                    if ts:
                        entry_time = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                        if entry_time < cutoff:
                            stale_entries.append(entry)
                except (ValueError, TypeError):
                    continue  # Unparseable timestamp — skip entry

        target_entries = junk_entries if junk_mode else stale_entries
        label = "junk" if junk_mode else f"older than {age_days} days"

        if not target_entries:
            print(f"✓ No unprocessed raw entries detected as {label}.")
            return

        print(f"Found {len(target_entries)} entries ({label}):\n")

        for entry in target_entries[:15]:  # Show max 15
            captured = entry.get("captured_at") or entry.get("timestamp") or ""
            captured_str = captured[:10] if captured else "unknown"
            blob = entry.get("blob") or entry.get("content") or ""
            blob_preview = blob[:50].replace("\n", " ")
            if len(blob) > 50:
                blob_preview += "..."
            print(f"  [{entry['id'][:8]}] {captured_str}: {blob_preview}")

        if len(target_entries) > 15:
            print(f"  ... and {len(target_entries) - 15} more")

        if dry_run:
            print(f"\n⚠ DRY RUN: Would delete {len(target_entries)} entries.")
            if junk_mode:
                print("  To actually delete, run: kernle raw clean --junk --confirm")
            else:
                print(f"  To actually delete, run: kernle raw clean --age {age_days} --confirm")
        else:
            deleted = 0
            for entry in target_entries:
                try:
                    k._storage.delete_raw(entry["id"])
                    deleted += 1
                except Exception as e:
                    print(f"  ✗ Failed to delete {entry['id'][:8]}: {e}")
            print(f"\n✓ Deleted {deleted} {label} raw entries.")

    elif args.raw_action == "promote":
        # Resolve the raw entry ID
        try:
            full_id = resolve_raw_id(k, args.id)
        except ValueError as e:
            print(f"✗ {e}")
            return

        target_type = args.type

        # Build kwargs for process_raw — only pass episode-specific args
        kwargs = {}
        if target_type == "episode":
            if args.objective:
                kwargs["objective"] = args.objective
            if args.outcome:
                kwargs["outcome"] = args.outcome

        try:
            result_id = k.process_raw(raw_id=full_id, as_type=target_type, **kwargs)
            print(f"✓ Promoted to {target_type}: {result_id[:8]}...")
            print("  Raw entry marked as processed.")
        except ValueError as e:
            print(f"✗ {e}")
            return

    elif args.raw_action == "triage":
        # Guided triage of unprocessed entries
        limit = getattr(args, "limit", 10)
        entries = k.list_raw(processed=False, limit=limit)

        if not entries:
            print("✓ No unprocessed raw entries to triage.")
            return

        print(f"Raw Entry Triage ({len(entries)} entries)")
        print("=" * 50)
        print()
        print("Suggestions: [E]pisode | [N]ote | [D]elete | [S]kip")
        print()

        for entry in entries:
            # Use blob (new) with content (legacy) as fallback
            blob = entry.get("blob") or entry.get("content") or ""
            captured = entry.get("captured_at") or entry.get("timestamp") or ""
            captured_str = captured[:16] if captured else "unknown"

            # Auto-suggest based on content analysis
            suggestion = "S"  # default skip
            content_lower = blob.lower()

            # Junk detection
            if len(blob.strip()) < 10 or content_lower in ["test", "list", "show", "help"]:
                suggestion = "D"
            # Session summaries / work logs → Episode
            elif any(
                x in content_lower
                for x in ["session", "completed", "shipped", "implemented", "built", "fixed"]
            ):
                suggestion = "E"
            # Insights / decisions → Note
            elif any(
                x in content_lower
                for x in ["insight", "decision", "realized", "learned", "important"]
            ):
                suggestion = "N"
            # Observations / patterns → Note (beliefs come from episode promotion)
            elif any(
                x in content_lower for x in ["believe", "think that", "seems like", "pattern"]
            ):
                suggestion = "N"

            suggestion_labels = {
                "E": "Episode",
                "N": "Note",
                "D": "Delete",
                "S": "Skip",
            }

            print(f"[{entry['id'][:8]}] {captured_str}")
            print(f"  {blob[:200]}{'...' if len(blob) > 200 else ''}")
            print(f"  → Suggested: {suggestion_labels[suggestion]}")
            print()
            print(f"  To act: kernle raw promote {entry['id'][:8]} --type <episode|note>")
            print("          kernle raw clean --junk --confirm  (to delete junk)")
            print("-" * 50)

    elif args.raw_action == "files":
        # Show flat file locations
        raw_dir = k._storage.get_raw_dir()
        files = k._storage.get_raw_files()

        print(f"Raw Flat Files Directory: {raw_dir}")
        print("=" * 50)

        if not files:
            print('\nNo raw files yet. Capture something with: kernle raw "thought"')
        else:
            print(f"\nFiles ({len(files)} total):")
            total_size = 0
            for f in files[:10]:
                size = f.stat().st_size
                total_size += size
                print(f"  {f.name:20} {size:>6} bytes")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more")
            print(f"\nTotal: {total_size:,} bytes")

        print("\n💡 Tips:")
        print(f"  • Edit directly: vim {raw_dir}/<date>.md")
        print(f"  • Search: grep -r 'pattern' {raw_dir}/")
        print(f"  • Git track: cd {raw_dir.parent} && git init")

        if getattr(args, "open", False):
            import subprocess

            subprocess.run(["open", str(raw_dir)], check=False)

    elif args.raw_action == "sync":
        # Sync from flat files to SQLite
        dry_run = getattr(args, "dry_run", False)

        if dry_run:
            print("DRY RUN: Scanning flat files for unindexed entries...")
        else:
            print("Syncing flat files to SQLite index...")

        result = k._storage.sync_raw_from_files()

        print(f"\nFiles processed: {result['files_processed']}")
        print(f"Entries imported: {result['imported']}")
        print(f"Entries skipped (already indexed): {result['skipped']}")

        if result["errors"]:
            print(f"\nErrors ({len(result['errors'])}):")
            for err in result["errors"][:5]:
                print(f"  • {err}")
            if len(result["errors"]) > 5:
                print(f"  ... and {len(result['errors']) - 5} more")

        if result["imported"] > 0:
            print(f"\n✓ Imported {result['imported']} entries from flat files")
        elif result["skipped"] > 0:
            print("\n✓ All entries already indexed")
        else:
            print("\n✓ No entries to import")
