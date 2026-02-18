"""Handlers for temporal tools: when, consolidate, consolidate_advanced, auto_capture."""

from collections import Counter
from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import (
    validate_enum,
    validate_number,
)

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_when(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["period"] = validate_enum(
        arguments.get("period"),
        "period",
        ["today", "yesterday", "this week", "last hour"],
        "today",
    )
    return sanitized


def validate_memory_consolidate(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["min_episodes"] = int(
        validate_number(arguments.get("min_episodes"), "min_episodes", 1, 100, 3)
    )
    return sanitized


def validate_memory_consolidate_advanced(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["episode_limit"] = int(
        validate_number(arguments.get("episode_limit"), "episode_limit", 1, 500, 100)
    )
    return sanitized


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_when(args: Dict[str, Any], k: Kernle) -> str:
    period = args.get("period", "today")
    temporal = k.what_happened(period)
    lines = [f"What happened {period}:\n"]
    if temporal.get("episodes"):
        lines.append("Episodes:")
        for ep in temporal["episodes"][:5]:
            lines.append(f"  - {ep['objective'][:60]} [{ep.get('outcome_type', '?')}]")
    if temporal.get("notes"):
        lines.append("Notes:")
        for n in temporal["notes"][:5]:
            lines.append(f"  - {n['content'][:60]}...")
    return "\n".join(lines)


def handle_memory_consolidate(args: Dict[str, Any], k: Kernle) -> str:
    min_episodes = args.get("min_episodes", 3)

    episodes = k._storage.get_episodes(limit=20)
    beliefs = k.load_beliefs(limit=15)

    lines = []
    lines.append("# Memory Consolidation: Reflection Scaffold")
    lines.append("")
    lines.append("Kernle has gathered your recent experiences and current beliefs.")
    lines.append("Your task: reason about patterns, extract insights, decide on belief updates.")
    lines.append("")

    lines.append("## Recent Experiences")
    lines.append("")
    if len(episodes) < min_episodes:
        lines.append(
            f"Only {len(episodes)} episode(s) recorded (minimum {min_episodes} for consolidation)."
        )
        lines.append("Continue capturing experiences before consolidating.")
    else:
        for i, ep in enumerate(episodes[:10], 1):
            outcome_emoji = {"success": "✓", "failure": "✗", "partial": "~"}.get(
                ep.outcome_type or "", "?"
            )
            lines.append(
                f"**{i}. {ep.objective}** [{outcome_emoji} {ep.outcome_type or 'unknown'}]"
            )
            lines.append(f"   Outcome: {ep.outcome}")
            if ep.lessons:
                for lesson in ep.lessons:
                    lines.append(f"   → Lesson: {lesson}")
            if ep.tags:
                lines.append(f"   Tags: {', '.join(ep.tags)}")
            lines.append("")

    lines.append("## Current Beliefs")
    lines.append("")
    if beliefs:
        for b in beliefs[:10]:
            conf = f" ({b['confidence']:.0%})" if b.get("confidence") else ""
            btype = f"[{b.get('belief_type', 'fact')}]" if b.get("belief_type") else ""
            lines.append(f"- {btype} {b['statement']}{conf}")
        lines.append("")
    else:
        lines.append("No beliefs recorded yet.")
        lines.append("")

    all_lessons = []
    for ep in episodes:
        if ep.lessons:
            all_lessons.extend(ep.lessons)

    if all_lessons:
        lesson_counts = Counter(all_lessons)
        recurring = [(lesson, cnt) for lesson, cnt in lesson_counts.items() if cnt >= 2]
        if recurring:
            lines.append("## Recurring Patterns")
            lines.append("")
            for lesson, count in sorted(recurring, key=lambda x: -x[1])[:5]:
                lines.append(f"- ({count}x) {lesson}")
            lines.append("")

    lines.append("---")
    lines.append("## Your Reflection Task")
    lines.append("")
    lines.append("Consider the experiences above and ask yourself:")
    lines.append("")
    lines.append("1. **Pattern Recognition**: What themes appear across multiple experiences?")
    lines.append("   - Are there repeated successes or failures?")
    lines.append("   - What approaches consistently work (or don't)?")
    lines.append("")
    lines.append("2. **Belief Validation**: Do your current beliefs match your experiences?")
    lines.append("   - Any beliefs that should increase in confidence?")
    lines.append("   - Any beliefs contradicted by recent outcomes?")
    lines.append("")
    lines.append("3. **New Insights**: What have you learned that isn't captured yet?")
    lines.append("   - Beliefs emerge from episodes — add experiences with `memory_episode`")
    lines.append("   - Update existing beliefs with `memory_belief_update`")
    lines.append("")
    lines.append("**Kernle provides the data. You do the reasoning.**")
    lines.append("")
    lines.append(f"Episodes reviewed: {len(episodes)} | Beliefs on file: {len(beliefs)}")

    return "\n".join(lines)


def handle_memory_consolidate_advanced(args: Dict[str, Any], k: Kernle) -> str:
    episode_limit = args.get("episode_limit", 100)
    advanced = k.scaffold_advanced_consolidation(episode_limit=episode_limit)
    return advanced["scaffold"]


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_when": handle_memory_when,
    "memory_consolidate": handle_memory_consolidate,
    "memory_consolidate_advanced": handle_memory_consolidate_advanced,
}

VALIDATORS = {
    "memory_when": validate_memory_when,
    "memory_consolidate": validate_memory_consolidate,
    "memory_consolidate_advanced": validate_memory_consolidate_advanced,
}
