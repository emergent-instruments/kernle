"""Hook commands for Claude Code integration.

These commands are invoked by Claude Code hooks configured via ``kernle setup claude-code``.
They read JSON from stdin, perform memory operations, and write JSON to stdout.

CRITICAL: All hook commands MUST exit 0 regardless of errors. A non-zero exit
from a hook breaks the Claude Code session.

Hook Failure Semantics
----------------------
Each hook type has an explicit failure mode that determines what happens when
an error occurs (validation failure, storage error, corrupted input, etc.):

- **PreToolUse: FAIL-CLOSED** -- Deny on any error, including validation.
  This is the security-critical hook that gates memory-file writes.  A silent
  pass-through on error would let unvalidated writes land on disk.

- **SessionStart: FAIL-OPEN** -- Return empty additionalContext on error.
  This is a lifecycle hook.  If memory cannot be loaded the session should
  still start; the user just won't see prior context.

- **PreCompact: FAIL-OPEN** -- Skip checkpoint and exit 0 on error.
  Missing a pre-compaction checkpoint is acceptable; blocking compaction
  would degrade the session.

- **SessionEnd: FAIL-OPEN** -- Skip checkpoint/capture and exit 0 on error.
  The session is already ending; there is no user-facing recovery path.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

MEMORY_PATTERNS = [
    re.compile(r"^memory/"),
    re.compile(r"/memory/"),
    re.compile(r"MEMORY\.md$"),
]


def _is_memory_path(file_path: str) -> bool:
    """Check if a file path targets a memory file."""
    return any(p.search(file_path) for p in MEMORY_PATTERNS)


def _resolve_hook_stack_id(
    explicit_stack: Optional[str],
    cwd: Optional[str] = None,
) -> str:
    """Resolve stack ID for hook context.

    Resolution order:
    1. explicit_stack (from --stack CLI flag, baked in by setup)
    2. KERNLE_STACK_ID environment variable
    3. cwd from stdin JSON (project directory name)
    4. Standard resolve_stack_id() auto-generation
    """
    if explicit_stack:
        return explicit_stack

    env_id = os.environ.get("KERNLE_STACK_ID")
    if env_id:
        return env_id

    if cwd:
        dir_name = Path(cwd).name
        if dir_name and dir_name not in ("workspace", "home", "/"):
            return dir_name

    from kernle.utils import resolve_stack_id

    return resolve_stack_id()


def _read_last_messages(
    transcript_path: Optional[str],
    max_user: int = 200,
    max_assistant: int = 500,
) -> Tuple[str, Optional[str]]:
    """Read last user and assistant messages from a transcript JSONL file.

    Returns (task, context) tuple suitable for checkpoint save.
    Falls back to generic text if transcript can't be read.
    """
    if not transcript_path:
        return ("Session ended", None)

    try:
        lines = Path(transcript_path).read_text().strip().split("\n")
        last_user = None
        last_assistant = None

        for line in reversed(lines):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # Malformed JSON line — skip

            role = entry.get("role", "")
            content = _extract_text(entry)
            if not content:
                continue

            if role == "user" and last_user is None:
                last_user = _truncate(content, max_user)
            elif role == "assistant" and last_assistant is None:
                last_assistant = _truncate(content, max_assistant)

            if last_user is not None and last_assistant is not None:
                break

        return (last_user or "Session ended", last_assistant)
    except Exception as exc:
        logger.debug(
            "Swallowed %s in _read_last_messages: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return ("Session ended", None)


def _extract_text(entry: dict) -> str:
    """Extract text content from a transcript entry.

    Content can be a string or a list of content blocks.
    """
    content = entry.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return " ".join(texts)
    return str(content)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


# -- Payload validation (#717) ------------------------------------------------

# Required keys per hook type.  Unknown hook types pass through for forward
# compatibility -- new hooks added upstream shouldn't break older kernle.
_REQUIRED_KEYS: dict[str, list[str]] = {
    "SessionStart": ["cwd"],
    "PreToolUse": ["cwd", "tool_input"],
    "PreCompact": ["cwd"],
    "SessionEnd": ["cwd"],
}


def _validate_hook_input(data: dict, hook_name: str) -> dict:
    """Validate that *data* contains the required keys for *hook_name*.

    Returns *data* unchanged on success.  Raises ``ValueError`` with a
    descriptive message when a required key is missing.

    Unknown hook names are passed through without validation so that newer
    Claude Code versions don't break older kernle installs.
    """
    required = _REQUIRED_KEYS.get(hook_name)
    if required is None:
        return data

    for key in required:
        if key not in data:
            raise ValueError(f"Hook '{hook_name}' requires key '{key}' in payload")

    return data


def cmd_hook_session_start(args) -> None:
    """SessionStart hook: Load memory and output as additionalContext.

    Failure mode: FAIL-OPEN -- return empty context on any error so the
    session can start without prior memory.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        _validate_hook_input(hook_input, "SessionStart")
        cwd = hook_input.get("cwd")
        stack_id = _resolve_hook_stack_id(getattr(args, "stack", None), cwd)

        budget = int(os.environ.get("KERNLE_TOKEN_BUDGET", "8000"))

        from kernle import Kernle

        k = Kernle(stack_id=stack_id)
        memory = k.load(budget=budget)
        formatted = k.format_memory(memory)

        if formatted:
            output = {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": formatted,
                }
            }
            json.dump(output, sys.stdout)
    except Exception as exc:
        # FAIL-OPEN: swallow all errors, produce no output
        logger.debug(
            "Swallowed %s in SessionStart hook: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    sys.exit(0)


def cmd_hook_pre_tool_use(args) -> None:
    """PreToolUse hook: Intercept and deny writes to memory files.

    Failure mode: FAIL-CLOSED -- emit deny on any error (validation,
    storage, import, etc.) so that unvalidated writes never land on disk.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        _validate_hook_input(hook_input, "PreToolUse")
        tool_input = hook_input.get("tool_input", {})

        file_path = tool_input.get("file_path", "") or tool_input.get("path", "")
        if not file_path or not _is_memory_path(file_path):
            sys.exit(0)

        # Capture content into Kernle as raw entry
        cwd = hook_input.get("cwd")
        stack_id = _resolve_hook_stack_id(getattr(args, "stack", None), cwd)

        from kernle import Kernle

        k = Kernle(stack_id=stack_id)

        content = tool_input.get("content", "") or tool_input.get("new_string", "")
        if content:
            truncated = content[:2000] + "\n[truncated]" if len(content) > 2000 else content
            k.raw(f"[memory-capture] {file_path}\n\n{truncated}", source="hook")

        # Block the write
        output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": (
                    "Memory writes are handled by Kernle. Use `kernle raw`, "
                    "`kernle episode`, or `kernle note` to record memories. "
                    "Loading and checkpointing are automatic."
                ),
            }
        }
        json.dump(output, sys.stdout)
    except Exception as e:
        # FAIL-CLOSED: emit deny using hookSpecificOutput schema
        # MUST use same schema as normal deny and exit(0) per hook contract
        logger.warning(
            "PreToolUse hook FAIL-CLOSED due to %s: %s",
            type(e).__name__,
            e,
            exc_info=True,
        )
        deny_output = {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": f"Hook internal error: {e}",
            }
        }
        try:
            json.dump(deny_output, sys.stdout)
        except Exception as exc2:
            logger.debug(
                "Swallowed %s writing deny output in PreToolUse: %s",
                type(exc2).__name__,
                exc2,
                exc_info=True,
            )
            # stdout itself is broken; still exit(0) per contract

    sys.exit(0)


def cmd_hook_pre_compact(args) -> None:
    """PreCompact hook: Save checkpoint before context compaction.

    Failure mode: FAIL-OPEN -- skip checkpoint and exit 0 on any error
    so that context compaction is never blocked.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        _validate_hook_input(hook_input, "PreCompact")
        cwd = hook_input.get("cwd")
        transcript_path = hook_input.get("transcript_path")
        stack_id = _resolve_hook_stack_id(getattr(args, "stack", None), cwd)

        from kernle import Kernle

        k = Kernle(stack_id=stack_id)
        task, context = _read_last_messages(transcript_path)
        k.checkpoint(f"[pre-compact] {task}", context=context)
    except Exception as exc:
        # FAIL-OPEN: swallow all errors, skip checkpoint
        logger.debug(
            "Swallowed %s in PreCompact hook: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    sys.exit(0)


def cmd_hook_session_end(args) -> None:
    """SessionEnd hook: Save checkpoint and raw entry on session termination.

    Failure mode: FAIL-OPEN -- skip operations and exit 0 on any error.
    The session is already ending; there is no recovery path.
    """
    try:
        hook_input = json.loads(sys.stdin.read())
        _validate_hook_input(hook_input, "SessionEnd")
        cwd = hook_input.get("cwd")
        transcript_path = hook_input.get("transcript_path")
        stack_id = _resolve_hook_stack_id(getattr(args, "stack", None), cwd)

        from kernle import Kernle

        k = Kernle(stack_id=stack_id)
        task, context = _read_last_messages(transcript_path)

        # Save both (ignore individual failures -- FAIL-OPEN per-operation)
        try:
            k.checkpoint(task, context=context)
        except Exception as exc:
            logger.debug(
                "Swallowed %s in SessionEnd checkpoint: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
        try:
            k.raw(f"Session ended. Task: {task}", source="hook")
        except Exception as exc:
            logger.debug(
                "Swallowed %s in SessionEnd raw capture: %s",
                type(exc).__name__,
                exc,
                exc_info=True,
            )
    except Exception as exc:
        # FAIL-OPEN: swallow all errors, skip all operations
        logger.debug(
            "Swallowed %s in SessionEnd hook: %s",
            type(exc).__name__,
            exc,
            exc_info=True,
        )

    sys.exit(0)


def cmd_hook(args) -> None:
    """Dispatch hook subcommands."""
    hook_event = getattr(args, "hook_event", None)
    if hook_event == "session-start":
        cmd_hook_session_start(args)
    elif hook_event == "pre-tool-use":
        cmd_hook_pre_tool_use(args)
    elif hook_event == "pre-compact":
        cmd_hook_pre_compact(args)
    elif hook_event == "session-end":
        cmd_hook_session_end(args)
    else:
        print("Usage: kernle hook {session-start|pre-tool-use|pre-compact|session-end}")
        sys.exit(0)
