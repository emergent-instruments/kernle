"""Belief revision commands for Kernle CLI."""

import json
from typing import TYPE_CHECKING

from kernle.cli.commands.helpers import validate_input

if TYPE_CHECKING:
    from kernle import Kernle


def cmd_belief(args, k: "Kernle"):
    """Handle belief revision subcommands."""
    if args.belief_action == "revise":
        episode_id = validate_input(args.episode_id, "episode_id", 100)
        result = k.revise_beliefs_from_episode(episode_id)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if result.get("error"):
                print(f"✗ {result['error']}")
                return

            print(f"Belief Revision from Episode {episode_id[:8]}...")
            print("=" * 50)

            # Reinforced beliefs
            reinforced = result.get("reinforced", [])
            if reinforced:
                print(f"\n✓ Reinforced ({len(reinforced)} beliefs):")
                for r in reinforced:
                    print(f"  • {r['statement'][:60]}...")
                    print(f"    ID: {r['belief_id'][:8]}...")

            # Contradicted beliefs
            contradicted = result.get("contradicted", [])
            if contradicted:
                print(f"\n⚠️  Potential Contradictions ({len(contradicted)}):")
                for c in contradicted:
                    print(f"  • {c['statement'][:60]}...")
                    print(f"    ID: {c['belief_id'][:8]}...")
                    print(f"    Evidence: {c['evidence'][:50]}...")

            # Suggested new beliefs
            suggested = result.get("suggested_new", [])
            if suggested:
                print(f"\n💡 Suggested New Beliefs ({len(suggested)}):")
                for s in suggested:
                    print(f"  • {s['statement'][:60]}...")
                    print(f"    Suggested confidence: {s['suggested_confidence']:.0%}")

            if not reinforced and not contradicted and not suggested:
                print("\nNo belief revisions found for this episode.")

    elif args.belief_action == "contradictions":
        statement = validate_input(args.statement, "statement", 2000)
        results = k.find_contradictions(statement, limit=args.limit)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            if not results:
                print(f'No contradictions found for: "{statement[:50]}..."')
                return

            print(f'Potential Contradictions for: "{statement[:50]}..."')
            print("=" * 60)

            for i, r in enumerate(results, 1):
                conf_bar = "█" * int(r["contradiction_confidence"] * 10) + "░" * (
                    10 - int(r["contradiction_confidence"] * 10)
                )
                status = "active" if r["is_active"] else "superseded"

                print(f"\n{i}. [{conf_bar}] {r['contradiction_confidence']:.0%} confidence")
                print(f"   Type: {r['contradiction_type']}")
                print(f"   Statement: {r['statement'][:60]}...")
                print(
                    f"   Belief ID: {r['belief_id'][:8]}... ({status}, reinforced {r['times_reinforced']}x)"
                )
                print(f"   Reason: {r['explanation']}")

    elif args.belief_action == "history":
        belief_id = validate_input(args.id, "belief_id", 100)
        history = k.get_belief_history(belief_id)

        if args.json:
            print(json.dumps(history, indent=2, default=str))
        else:
            if not history:
                print(f"No history found for belief {belief_id[:8]}...")
                return

            print("Belief Revision History")
            print("=" * 60)

            for i, entry in enumerate(history):
                is_current = ">>> " if entry["is_current"] else "    "
                status = "\U0001f7e2 active" if entry["is_active"] else "\u26ab revised"
                conf_bar = "\u2588" * int(entry["confidence"] * 5) + "\u2591" * (
                    5 - int(entry["confidence"] * 5)
                )

                print(f"\n{is_current}[{i + 1}] {entry['id'][:8]}... ({status})")
                print(f"     Statement: {entry['statement'][:55]}...")
                print(
                    f"     Confidence: [{conf_bar}] {entry['confidence']:.0%} | Reinforced: {entry['times_reinforced']}x"
                )
                print(
                    f"     Created: {entry['created_at'][:10] if entry['created_at'] else 'unknown'}"
                )

                if entry.get("supersession_reason"):
                    print(f"     Reason: {entry['supersession_reason'][:50]}...")

                # Show chain link if present (legacy data)
                if entry.get("superseded_by"):
                    print(f"     \u2192 Revised to: {entry['superseded_by'][:8]}...")

    elif args.belief_action == "reinforce":
        belief_id = validate_input(args.id, "belief_id", 100)
        evidence = getattr(args, "evidence", None)
        reason = getattr(args, "reason", None)

        if k.reinforce_belief(belief_id, evidence_source=evidence, reason=reason):
            print(f"✓ Belief {belief_id[:8]}... reinforced")
            if evidence:
                print(f"  Evidence: {evidence}")
        else:
            print(f"✗ Belief {belief_id[:8]}... not found")

    elif args.belief_action == "supersede":
        old_id = validate_input(args.old_id, "old_id", 100)
        new_statement = validate_input(args.new_statement, "new_statement", 2000)

        try:
            new_id = k.revise_belief(
                old_id=old_id,
                new_statement=new_statement,
                confidence=args.confidence,
                reason=args.reason,
            )
            print("✓ Belief revised")
            print(f"  Old: {old_id[:8]}... (now inactive)")
            print(f"  New: {new_id[:8]}... (active)")
            print(f"  Statement: {new_statement[:60]}...")
            print(f"  Confidence: {args.confidence:.0%}")
        except ValueError as e:
            print(f"✗ {e}")

    elif args.belief_action == "list":
        # Get beliefs from storage directly for more detail
        beliefs = k._storage.get_beliefs(limit=args.limit, include_inactive=args.all)

        # Client-side filtering by scope/domain/abstraction (KEP v3)
        scope_filter = getattr(args, "scope", None)
        domain_filter = getattr(args, "domain", None)
        abstraction_filter = getattr(args, "abstraction_level", None)
        if scope_filter:
            beliefs = [b for b in beliefs if getattr(b, "belief_scope", "world") == scope_filter]
        if domain_filter:
            beliefs = [b for b in beliefs if getattr(b, "source_domain", None) == domain_filter]
        if abstraction_filter:
            beliefs = [
                b
                for b in beliefs
                if getattr(b, "abstraction_level", "specific") == abstraction_filter
            ]

        if args.json:
            data = [
                {
                    "id": b.id,
                    "statement": b.statement,
                    "confidence": b.confidence,
                    "times_reinforced": b.times_reinforced,
                    "is_active": b.is_active,
                    "supersedes": b.supersedes,
                    "superseded_by": b.superseded_by,
                    "belief_scope": getattr(b, "belief_scope", "world"),
                    "source_domain": getattr(b, "source_domain", None),
                    "cross_domain_applications": getattr(b, "cross_domain_applications", None),
                    "abstraction_level": getattr(b, "abstraction_level", "specific"),
                    "created_at": b.created_at.isoformat() if b.created_at else None,
                }
                for b in beliefs
            ]
            print(json.dumps(data, indent=2, default=str))
        else:
            active_count = sum(1 for b in beliefs if b.is_active)
            print(f"Beliefs ({len(beliefs)} total, {active_count} active)")
            print("=" * 60)

            for b in beliefs:
                status = "\U0001f7e2" if b.is_active else "\u26ab"
                conf_bar = "\u2588" * int(b.confidence * 5) + "\u2591" * (5 - int(b.confidence * 5))
                reinf = f"(+{b.times_reinforced})" if b.times_reinforced > 0 else ""
                scope = getattr(b, "belief_scope", "world")
                scope_tag = f" [{scope}]" if scope != "world" else ""

                print(f"\n{status} [{conf_bar}] {b.confidence:.0%} {reinf}{scope_tag}")
                print(f"   {b.statement[:60]}{'...' if len(b.statement) > 60 else ''}")
                print(f"   ID: {b.id[:8]}...")

                domain = getattr(b, "source_domain", None)
                if domain:
                    print(f"   Domain: {domain}")

                if b.supersedes:
                    print(f"   Supersedes: {b.supersedes[:8]}...")
                if b.superseded_by:
                    print(f"   Superseded by: {b.superseded_by[:8]}...")
