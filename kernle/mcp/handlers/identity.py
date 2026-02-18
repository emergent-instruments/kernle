"""Handlers for identity tools: value, goal, drive + list/update variants (belief list/update only)."""

import json
from typing import Any, Dict

from kernle.core import Kernle
from kernle.mcp.sanitize import (
    sanitize_array,
    sanitize_source_metadata,
    sanitize_string,
    validate_enum,
    validate_number,
)
from kernle.mcp.tool_definitions import VALID_SOURCE_TYPES

# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_memory_value(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["name"] = sanitize_string(arguments.get("name"), "name", 100, required=True)
    sanitized["statement"] = sanitize_string(
        arguments.get("statement"), "statement", 1000, required=True
    )
    sanitized["priority"] = int(validate_number(arguments.get("priority"), "priority", 0, 100, 50))
    sanitized.update(sanitize_source_metadata(arguments, VALID_SOURCE_TYPES))
    return sanitized


def validate_memory_goal(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["title"] = sanitize_string(arguments.get("title"), "title", 200, required=True)
    sanitized["description"] = sanitize_string(
        arguments.get("description"), "description", 1000, required=False
    )
    sanitized["priority"] = validate_enum(
        arguments.get("priority"), "priority", ["low", "medium", "high"], "medium"
    )
    sanitized.update(sanitize_source_metadata(arguments, VALID_SOURCE_TYPES))
    return sanitized


def validate_memory_drive(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["drive_type"] = validate_enum(
        arguments.get("drive_type"),
        "drive_type",
        ["existence", "growth", "curiosity", "connection", "reproduction"],
        default=None,
        required=True,
    )
    sanitized["intensity"] = validate_number(arguments.get("intensity"), "intensity", 0.0, 1.0, 0.5)
    sanitized["focus_areas"] = sanitize_array(arguments.get("focus_areas"), "focus_areas", 200, 10)
    sanitized.update(sanitize_source_metadata(arguments, VALID_SOURCE_TYPES))
    return sanitized


def validate_memory_belief_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 20))
    sanitized["format"] = validate_enum(arguments.get("format"), "format", ["text", "json"], "text")
    return sanitized


def validate_memory_value_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
    sanitized["format"] = validate_enum(arguments.get("format"), "format", ["text", "json"], "text")
    return sanitized


def validate_memory_goal_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["status"] = validate_enum(
        arguments.get("status"),
        "status",
        ["active", "completed", "paused", "all"],
        "active",
    )
    sanitized["limit"] = int(validate_number(arguments.get("limit"), "limit", 1, 100, 10))
    sanitized["format"] = validate_enum(arguments.get("format"), "format", ["text", "json"], "text")
    return sanitized


def validate_memory_drive_list(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["format"] = validate_enum(arguments.get("format"), "format", ["text", "json"], "text")
    return sanitized


def validate_memory_goal_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["goal_id"] = sanitize_string(arguments.get("goal_id"), "goal_id", 100, required=True)
    sanitized["status"] = (
        validate_enum(arguments.get("status"), "status", ["active", "completed", "paused"], None)
        if arguments.get("status")
        else None
    )
    sanitized["priority"] = (
        validate_enum(arguments.get("priority"), "priority", ["low", "medium", "high"], None)
        if arguments.get("priority")
        else None
    )
    sanitized["description"] = sanitize_string(
        arguments.get("description"), "description", 1000, required=False
    )
    return sanitized


def validate_memory_belief_update(arguments: Dict[str, Any]) -> Dict[str, Any]:
    sanitized: Dict[str, Any] = {}
    sanitized["belief_id"] = sanitize_string(
        arguments.get("belief_id"), "belief_id", 100, required=True
    )
    sanitized["confidence"] = (
        validate_number(arguments.get("confidence"), "confidence", 0.0, 1.0, None)
        if arguments.get("confidence") is not None
        else None
    )
    sanitized["is_active"] = arguments.get("is_active")  # Boolean, can be None
    return sanitized


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_memory_value(args: Dict[str, Any], k: Kernle) -> str:
    k.value(
        name=args["name"],
        statement=args["statement"],
        priority=args.get("priority", 50),
        context=args.get("context"),
        context_tags=args.get("context_tags"),
        source=args.get("source"),
        derived_from=args.get("derived_from"),
        source_type=args.get("source_type"),
    )
    return f"Value saved: {args['name']}"


def handle_memory_goal(args: Dict[str, Any], k: Kernle) -> str:
    k.goal(
        title=args["title"],
        description=args.get("description"),
        priority=args.get("priority", "medium"),
        context=args.get("context"),
        context_tags=args.get("context_tags"),
        source=args.get("source"),
        derived_from=args.get("derived_from"),
        source_type=args.get("source_type"),
    )
    return f"Goal saved: {args['title']}"


def handle_memory_drive(args: Dict[str, Any], k: Kernle) -> str:
    k.drive(
        drive_type=args["drive_type"],
        intensity=args.get("intensity", 0.5),
        focus_areas=args.get("focus_areas"),
        context=args.get("context"),
        context_tags=args.get("context_tags"),
        source=args.get("source"),
        derived_from=args.get("derived_from"),
        source_type=args.get("source_type"),
    )
    return f"Drive '{args['drive_type']}' set to {args.get('intensity', 0.5):.0%}"


def handle_memory_belief_list(args: Dict[str, Any], k: Kernle) -> str:
    beliefs = k.load_beliefs(limit=args.get("limit", 20))
    format_type = args.get("format", "text")
    if not beliefs:
        return "No beliefs found." if format_type == "text" else json.dumps([], indent=2)
    if format_type == "json":
        return json.dumps(beliefs, indent=2, default=str)
    lines = [f"Found {len(beliefs)} belief(s):\n"]
    for i, b in enumerate(beliefs, 1):
        conf = f" ({b['confidence']:.0%})" if b.get("confidence") else ""
        btype = f"[{b.get('belief_type', 'fact')}]" if b.get("belief_type") else ""
        lines.append(f"{i}. {btype} {b['statement']}{conf}")
    return "\n".join(lines)


def handle_memory_value_list(args: Dict[str, Any], k: Kernle) -> str:
    values = k.load_values(limit=args.get("limit", 10))
    format_type = args.get("format", "text")
    if not values:
        return "No values found." if format_type == "text" else json.dumps([], indent=2)
    if format_type == "json":
        return json.dumps(values, indent=2, default=str)
    lines = [f"Found {len(values)} value(s):\n"]
    for i, v in enumerate(values, 1):
        priority = f" (priority: {v.get('priority', 50)})" if v.get("priority") else ""
        lines.append(f"{i}. **{v['name']}**: {v['statement']}{priority}")
    return "\n".join(lines)


def handle_memory_goal_list(args: Dict[str, Any], k: Kernle) -> str:
    status = args.get("status", "active")
    format_type = args.get("format", "text")
    goals = k.load_goals(limit=args.get("limit", 10), status=status)

    if not goals:
        return f"No {status} goals found." if format_type == "text" else json.dumps([], indent=2)
    if format_type == "json":
        return json.dumps(goals, indent=2, default=str)
    lines = [f"Found {len(goals)} goal(s):\n"]
    for i, g in enumerate(goals, 1):
        priority = f" [{g.get('priority', 'medium')}]" if g.get("priority") else ""
        status_str = f" ({g.get('status', 'active')})" if g.get("status") != "active" else ""
        lines.append(f"{i}. {g['title']}{priority}{status_str}")
        if g.get("description"):
            lines.append(f"   {g['description'][:60]}...")
    return "\n".join(lines)


def handle_memory_drive_list(args: Dict[str, Any], k: Kernle) -> str:
    drives = k.load_drives()
    format_type = args.get("format", "text")
    if not drives:
        return "No drives configured." if format_type == "text" else json.dumps([], indent=2)
    if format_type == "json":
        return json.dumps(drives, indent=2, default=str)
    lines = ["Current drives:\n"]
    for d in drives:
        focus = f" → {', '.join(d.get('focus_areas', []))}" if d.get("focus_areas") else ""
        lines.append(f"- **{d['drive_type']}**: {d['intensity']:.0%}{focus}")
    return "\n".join(lines)


def handle_memory_goal_update(args: Dict[str, Any], k: Kernle) -> str:
    goal_id = args["goal_id"]
    updated = k.update_goal(
        goal_id=goal_id,
        status=args.get("status"),
        priority=args.get("priority"),
        description=args.get("description"),
    )
    if updated:
        return f"Goal {goal_id[:8]}... updated successfully."
    return f"Goal {goal_id[:8]}... not found."


def handle_memory_belief_update(args: Dict[str, Any], k: Kernle) -> str:
    belief_id = args["belief_id"]
    updated = k.update_belief(
        belief_id=belief_id,
        confidence=args.get("confidence"),
        is_active=args.get("is_active"),
    )
    if updated:
        return f"Belief {belief_id[:8]}... updated successfully."
    return f"Belief {belief_id[:8]}... not found."


# ---------------------------------------------------------------------------
# Registry dicts
# ---------------------------------------------------------------------------

HANDLERS = {
    "memory_value": handle_memory_value,
    "memory_goal": handle_memory_goal,
    "memory_drive": handle_memory_drive,
    "memory_belief_list": handle_memory_belief_list,
    "memory_value_list": handle_memory_value_list,
    "memory_goal_list": handle_memory_goal_list,
    "memory_drive_list": handle_memory_drive_list,
    "memory_goal_update": handle_memory_goal_update,
    "memory_belief_update": handle_memory_belief_update,
}

VALIDATORS = {
    "memory_value": validate_memory_value,
    "memory_goal": validate_memory_goal,
    "memory_drive": validate_memory_drive,
    "memory_belief_list": validate_memory_belief_list,
    "memory_value_list": validate_memory_value_list,
    "memory_goal_list": validate_memory_goal_list,
    "memory_drive_list": validate_memory_drive_list,
    "memory_goal_update": validate_memory_goal_update,
    "memory_belief_update": validate_memory_belief_update,
}
