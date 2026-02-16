"""Doctor command for Kernle CLI - validates boot sequence compliance and system health."""

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    from kernle import Kernle

# Import seed beliefs version for comparison
from kernle.cli.commands.migrate import (
    _FULL_SEED_BELIEFS,
    _MINIMAL_SEED_BELIEFS,
    SEED_BELIEFS_VERSION,
)

logger = logging.getLogger(__name__)


class ComplianceCheck:
    """Result of a single compliance check."""

    def __init__(
        self,
        name: str,
        passed: bool,
        message: str,
        fix: Optional[str] = None,
        category: str = "required",
    ):
        self.name = name
        self.passed = passed
        self.message = message
        self.fix = fix
        self.category = category  # "required", "recommended", "info"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "message": self.message,
            "fix": self.fix,
            "category": self.category,
        }


def find_instruction_file() -> Optional[Tuple[Path, str]]:
    """Find instruction file and return (path, type)."""
    candidates = [
        (Path("CLAUDE.md"), "claude"),
        (Path("AGENTS.md"), "agents"),
        (Path(".cursorrules"), "cursor"),
        (Path(".clinerules"), "cline"),
        (Path.home() / ".claude" / "CLAUDE.md", "claude-global"),
    ]

    for path, file_type in candidates:
        if path.exists():
            return path, file_type

    return None


def check_kernle_load(content: str, stack_id: str) -> ComplianceCheck:
    """Check if load instruction is present."""
    patterns = [
        rf"kernle\s+(-[sa]\s+{re.escape(stack_id)}\s+)?load",
        r"kernle\s+-[sa]\s+\w+\s+load",
        r"kernle_load",  # MCP tool name
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="load_instruction", passed=True, message="✓ Load instruction found"
            )

    return ComplianceCheck(
        name="load_instruction",
        passed=False,
        message="✗ Missing `kernle load` instruction at session start",
        fix=f"Add: `kernle -s {stack_id} load` to session boot sequence",
    )


def check_kernle_anxiety(content: str, stack_id: str) -> ComplianceCheck:
    """Check if anxiety/health check instruction is present."""
    patterns = [
        rf"kernle\s+(-[sa]\s+{re.escape(stack_id)}\s+)?anxiety",
        r"kernle\s+-[sa]\s+\w+\s+anxiety",
        r"kernle_anxiety",  # MCP tool name
        r"health\s*check",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="anxiety_instruction", passed=True, message="✓ Health check instruction found"
            )

    return ComplianceCheck(
        name="anxiety_instruction",
        passed=False,
        message="✗ Missing `kernle anxiety` health check instruction",
        fix=f"Add: `kernle -s {stack_id} anxiety` after load",
    )


def check_per_message_health(content: str, stack_id: str) -> ComplianceCheck:
    """Check if per-message health check instruction is present."""
    patterns = [
        r"every\s+message",
        r"per.?message",
        r"before\s+(processing|any)\s+request",
        r"health\s+check.+message",
        r"anxiety\s+-b",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="per_message_health",
                passed=True,
                message="✓ Per-message health check instruction found",
            )

    return ComplianceCheck(
        name="per_message_health",
        passed=False,
        message="⚠ No per-message health check (recommended)",
        fix=f"Add section: 'Every Message: `kernle -s {stack_id} anxiety -b`'",
    )


def check_checkpoint_instruction(content: str, stack_id: str) -> ComplianceCheck:
    """Check if checkpoint instruction is present."""
    patterns = [
        rf"kernle\s+(-[sa]\s+{re.escape(stack_id)}\s+)?checkpoint",
        r"kernle\s+-[sa]\s+\w+\s+checkpoint",
        r"kernle_checkpoint",  # MCP tool name
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="checkpoint_instruction", passed=True, message="✓ Checkpoint instruction found"
            )

    return ComplianceCheck(
        name="checkpoint_instruction",
        passed=False,
        message="⚠ No checkpoint instruction (recommended for session end)",
        fix=f'Add: `kernle -s {stack_id} checkpoint save "state"` before session ends',
    )


def check_memory_section(content: str) -> ComplianceCheck:
    """Check if there's a dedicated memory section."""
    patterns = [
        r"##\s*Memory",
        r"##\s*Kernle",
        r"##\s*Every\s+Session",
        r"##\s*Boot\s*Sequence",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return ComplianceCheck(
                name="memory_section", passed=True, message="✓ Dedicated memory section found"
            )

    return ComplianceCheck(
        name="memory_section",
        passed=False,
        message="⚠ No dedicated Memory/Kernle section (recommended for clarity)",
        fix="Add: `## Memory (Kernle)` section header",
    )


def run_all_checks(content: str, stack_id: str) -> List[ComplianceCheck]:
    """Run all instruction file compliance checks."""
    checks = [
        check_memory_section(content),
        check_kernle_load(content, stack_id),
        check_kernle_anxiety(content, stack_id),
        check_per_message_health(content, stack_id),
        check_checkpoint_instruction(content, stack_id),
    ]
    return checks


# =============================================================================
# Seed Beliefs Checks
# =============================================================================


def check_seed_beliefs(k: "Kernle") -> Tuple[ComplianceCheck, dict]:
    """Check if agent has foundational seed beliefs.

    Returns:
        (check_result, details) where details contains counts and missing beliefs
    """
    try:
        existing = k._storage.get_beliefs(limit=200, include_inactive=False)
        existing_statements = {b.statement for b in existing}

        # Count how many seed beliefs exist
        full_count = sum(1 for b in _FULL_SEED_BELIEFS if b["statement"] in existing_statements)
        minimal_count = sum(
            1 for b in _MINIMAL_SEED_BELIEFS if b["statement"] in existing_statements
        )

        # Find missing beliefs
        missing_minimal = [
            b for b in _MINIMAL_SEED_BELIEFS if b["statement"] not in existing_statements
        ]
        missing_full = [b for b in _FULL_SEED_BELIEFS if b["statement"] not in existing_statements]

        details = {
            "total_beliefs": len(existing),
            "seed_beliefs_full": full_count,
            "seed_beliefs_minimal": minimal_count,
            "full_total": len(_FULL_SEED_BELIEFS),
            "minimal_total": len(_MINIMAL_SEED_BELIEFS),
            "missing_minimal": len(missing_minimal),
            "missing_full": len(missing_full),
            "version": SEED_BELIEFS_VERSION,
        }

        # Meta-belief is the most important - check specifically
        meta_belief = _MINIMAL_SEED_BELIEFS[0]  # The meta-belief is first
        has_meta = meta_belief["statement"] in existing_statements

        if full_count == len(_FULL_SEED_BELIEFS):
            return (
                ComplianceCheck(
                    name="seed_beliefs",
                    passed=True,
                    message=f"✓ Full seed beliefs present ({full_count}/{len(_FULL_SEED_BELIEFS)}) v{SEED_BELIEFS_VERSION}",
                    category="recommended",
                ),
                details,
            )
        elif minimal_count == len(_MINIMAL_SEED_BELIEFS):
            return (
                ComplianceCheck(
                    name="seed_beliefs",
                    passed=True,
                    message=f"✓ Minimal seed beliefs present ({minimal_count}/{len(_MINIMAL_SEED_BELIEFS)}), {len(missing_full)} optional missing",
                    category="recommended",
                ),
                details,
            )
        elif has_meta:
            return (
                ComplianceCheck(
                    name="seed_beliefs",
                    passed=True,
                    message=f"⚠ Partial seed beliefs ({full_count}/{len(_FULL_SEED_BELIEFS)}), meta-belief present",
                    fix="Run: kernle migrate seed-beliefs minimal",
                    category="recommended",
                ),
                details,
            )
        else:
            return (
                ComplianceCheck(
                    name="seed_beliefs",
                    passed=False,
                    message=f"⚠ No seed beliefs found ({full_count}/{len(_FULL_SEED_BELIEFS)})",
                    fix="Run: kernle migrate seed-beliefs",
                    category="recommended",
                ),
                details,
            )

    except Exception as e:
        logger.debug("Seed beliefs compliance check failed: %s", e, exc_info=True)
        return (
            ComplianceCheck(
                name="seed_beliefs",
                passed=False,
                message=f"✗ Could not check beliefs: {e}",
                category="info",
            ),
            {"error": str(e)},
        )


# =============================================================================
# Hook Installation Checks
# =============================================================================


def detect_platform() -> str:
    """Detect which platform we're running on based on available configs."""
    # Check for OpenClaw
    openclaw_config = Path.home() / ".openclaw" / "openclaw.json"
    if openclaw_config.exists():
        return "openclaw"

    # Check for Claude Code
    claude_global = Path.home() / ".claude" / "settings.json"
    claude_local = Path.cwd() / ".claude" / "settings.json"
    if claude_global.exists() or claude_local.exists():
        return "claude-code"

    # Default
    return "unknown"


def check_openclaw_hook() -> ComplianceCheck:
    """Check if OpenClaw kernle-load hook is installed and enabled."""
    # Check if hook files exist
    user_hooks = Path.home() / ".config" / "openclaw" / "hooks" / "kernle-load"
    bundled_hooks = Path.home() / "openclaw" / "src" / "hooks" / "bundled" / "kernle-load"

    hook_installed = user_hooks.exists() or bundled_hooks.exists()
    hook_location = "bundled" if bundled_hooks.exists() else "user" if user_hooks.exists() else None

    if not hook_installed:
        return ComplianceCheck(
            name="openclaw_hook",
            passed=False,
            message="✗ OpenClaw hook not installed",
            fix="Run: kernle setup openclaw",
            category="recommended",
        )

    # Check if enabled in config
    config_path = Path.home() / ".openclaw" / "openclaw.json"
    if not config_path.exists():
        return ComplianceCheck(
            name="openclaw_hook",
            passed=False,
            message=f"⚠ Hook installed ({hook_location}) but openclaw.json not found",
            fix="Create ~/.openclaw/openclaw.json with hook config",
            category="recommended",
        )

    try:
        with open(config_path) as f:
            config = json.load(f)

        enabled = (
            config.get("hooks", {})
            .get("internal", {})
            .get("entries", {})
            .get("kernle-load", {})
            .get("enabled", False)
        )

        if enabled:
            return ComplianceCheck(
                name="openclaw_hook",
                passed=True,
                message=f"✓ OpenClaw hook installed ({hook_location}) and enabled",
                category="recommended",
            )
        else:
            return ComplianceCheck(
                name="openclaw_hook",
                passed=False,
                message=f"⚠ Hook installed ({hook_location}) but NOT enabled in config",
                fix="Run: kernle setup openclaw --enable",
                category="recommended",
            )
    except Exception as e:
        logger.debug("OpenClaw hook config check failed: %s", e, exc_info=True)
        return ComplianceCheck(
            name="openclaw_hook",
            passed=False,
            message=f"⚠ Hook installed but could not read config: {e}",
            fix="Check ~/.openclaw/openclaw.json",
            category="recommended",
        )


def check_claude_code_hook(stack_id: str) -> ComplianceCheck:
    """Check if Claude Code hooks are configured (all 4 events)."""
    global_config = Path.home() / ".claude" / "settings.json"
    local_config = Path.cwd() / ".claude" / "settings.json"

    required_events = {"SessionStart", "PreToolUse", "PreCompact", "SessionEnd"}

    for config_path, location in [(global_config, "global"), (local_config, "project")]:
        if not config_path.exists():
            continue

        try:
            with open(config_path) as f:
                config = json.load(f)

            existing_hooks = config.get("hooks", {})
            configured_events = set()
            for event in required_events:
                hooks_list = existing_hooks.get(event, [])
                if any("kernle" in str(h).lower() for h in hooks_list):
                    configured_events.add(event)

            if configured_events == required_events:
                return ComplianceCheck(
                    name="claude_code_hook",
                    passed=True,
                    message=f"✓ Claude Code hooks configured ({location}, 4/4 events)",
                    category="recommended",
                )
            elif configured_events:
                missing = required_events - configured_events
                return ComplianceCheck(
                    name="claude_code_hook",
                    passed=False,
                    message=f"Claude Code hooks partially configured ({location}, {len(configured_events)}/4)",
                    fix=f"Run: kernle setup claude-code --force  (missing: {', '.join(sorted(missing))})",
                    category="recommended",
                )
        except Exception as e:
            logger.debug(f"Failed to read config at {config_path}: {e}", exc_info=True)
            continue

    return ComplianceCheck(
        name="claude_code_hook",
        passed=False,
        message="Claude Code hooks not configured",
        fix="Run: kernle setup claude-code",
        category="recommended",
    )


def check_hooks(stack_id: str) -> List[ComplianceCheck]:
    """Check hook installation based on detected platform."""
    platform = detect_platform()
    checks = []

    if platform == "openclaw":
        checks.append(check_openclaw_hook())
    elif platform == "claude-code":
        checks.append(check_claude_code_hook(stack_id))
    else:
        # Check both if platform unknown
        openclaw_check = check_openclaw_hook()
        claude_check = check_claude_code_hook(stack_id)

        # Only report as failed if BOTH are not installed
        if not openclaw_check.passed and not claude_check.passed:
            checks.append(
                ComplianceCheck(
                    name="hooks",
                    passed=False,
                    message="⚠ No platform hooks detected",
                    fix="Run: kernle setup openclaw (or claude-code)",
                    category="recommended",
                )
            )
        elif openclaw_check.passed:
            checks.append(openclaw_check)
        else:
            checks.append(claude_check)

    return checks


def cmd_doctor(args, k: "Kernle"):
    """Validate Kernle setup: instructions, beliefs, and hooks.

    Checks:
    1. Instruction file (CLAUDE.md, AGENTS.md, etc.) for boot sequence
    2. Seed beliefs presence and version
    3. Platform hooks (OpenClaw/Claude Code) installation

    Use --full for comprehensive check including beliefs and hooks.
    """
    stack_id = k.stack_id
    output_json = getattr(args, "json", False)
    fix = getattr(args, "fix", False)
    full_check = getattr(args, "full", False)

    all_checks: List[ComplianceCheck] = []
    belief_details = {}

    # -------------------------------------------------------------------------
    # Section 1: Instruction File Checks
    # -------------------------------------------------------------------------
    result = find_instruction_file()
    file_path = None
    file_type = None
    no_file = False

    if result is None:
        no_file = True
        all_checks.append(
            ComplianceCheck(
                name="instruction_file",
                passed=False,
                message="✗ No instruction file found (CLAUDE.md, AGENTS.md)",
                fix="Run: kernle init",
                category="required",
            )
        )
    else:
        file_path, file_type = result
        content = file_path.read_text()
        all_checks.append(
            ComplianceCheck(
                name="instruction_file",
                passed=True,
                message=f"✓ Instruction file found: {file_path.name}",
                category="required",
            )
        )
        # Run instruction content checks
        all_checks.extend(run_all_checks(content, stack_id))

    # -------------------------------------------------------------------------
    # Section 2: Seed Beliefs Checks (always run in --full mode)
    # -------------------------------------------------------------------------
    if full_check:
        belief_check, belief_details = check_seed_beliefs(k)
        all_checks.append(belief_check)

    # -------------------------------------------------------------------------
    # Section 3: Hook Checks (always run in --full mode)
    # -------------------------------------------------------------------------
    if full_check:
        hook_checks = check_hooks(stack_id)
        all_checks.extend(hook_checks)

    # -------------------------------------------------------------------------
    # Calculate summary
    # -------------------------------------------------------------------------
    required_names = ["instruction_file", "load_instruction", "anxiety_instruction"]
    recommended_names = [
        "per_message_health",
        "checkpoint_instruction",
        "memory_section",
        "seed_beliefs",
        "openclaw_hook",
        "claude_code_hook",
        "hooks",
    ]

    required_checks = [c for c in all_checks if c.name in required_names]
    recommended_checks = [c for c in all_checks if c.name in recommended_names]

    required_passed = sum(1 for c in required_checks if c.passed)
    required_total = len(required_checks)
    recommended_passed = sum(1 for c in recommended_checks if c.passed)
    recommended_total = len(recommended_checks)

    all_required_pass = required_passed == required_total

    # Determine overall status
    if no_file:
        status = "no_file"
        status_emoji = "❌"
        status_message = "No instruction file found"
    elif all_required_pass and recommended_passed == recommended_total:
        status = "excellent"
        status_emoji = "🟢"
        status_message = "Excellent! Full compliance"
    elif all_required_pass:
        status = "good"
        status_emoji = "🟡"
        status_message = "Good - required checks pass, some recommendations missing"
    else:
        status = "needs_work"
        status_emoji = "🔴"
        status_message = "Needs work - missing required components"

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------
    if output_json:
        output = {
            "status": status,
            "file": str(file_path) if file_path else None,
            "file_type": file_type,
            "stack_id": stack_id,
            "seed_beliefs_version": SEED_BELIEFS_VERSION,
            "required_passed": required_passed,
            "required_total": required_total,
            "recommended_passed": recommended_passed,
            "recommended_total": recommended_total,
            "checks": [c.to_dict() for c in all_checks],
            "belief_details": belief_details if full_check else None,
        }
        print(json.dumps(output, indent=2))
        return

    # Print header
    print()
    print("╔══════════════════════════════════════════════════╗")
    print("║         Kernle Doctor - System Health            ║")
    print("╚══════════════════════════════════════════════════╝")
    print(f"  Stack: {stack_id}")
    print(f"  Seed Beliefs Version: {SEED_BELIEFS_VERSION}")
    if file_path:
        print(f"  Instruction File: {file_path}")
    print()

    # Print status
    print(f"{status_emoji} Status: {status_message}")
    print(f"   Required: {required_passed}/{required_total}")
    print(f"   Recommended: {recommended_passed}/{recommended_total}")
    print()

    # Group checks by category
    print("─" * 50)
    print("INSTRUCTION FILE CHECKS")
    print("─" * 50)
    instruction_checks = [
        c
        for c in all_checks
        if c.name in ["instruction_file", *required_names[1:], *recommended_names[:3]]
    ]
    for check in instruction_checks:
        is_required = check.name in required_names
        prefix = "[required]    " if is_required else "[recommended] "
        print(f"  {prefix} {check.message}")

    if full_check:
        print()
        print("─" * 50)
        print("SEED BELIEFS")
        print("─" * 50)
        belief_checks = [c for c in all_checks if c.name == "seed_beliefs"]
        for check in belief_checks:
            print(f"  [recommended]  {check.message}")
        if belief_details and "total_beliefs" in belief_details:
            print(f"                  Total beliefs: {belief_details['total_beliefs']}")

        print()
        print("─" * 50)
        print("PLATFORM HOOKS")
        print("─" * 50)
        hook_checks = [c for c in all_checks if "hook" in c.name]
        if hook_checks:
            for check in hook_checks:
                print(f"  [recommended]  {check.message}")
        else:
            print("  [info]         No platform detected")

    # Print fixes if needed
    failed_checks = [c for c in all_checks if not c.passed and c.fix]
    if failed_checks:
        print()
        print("─" * 50)
        print("SUGGESTED FIXES")
        print("─" * 50)
        for check in failed_checks:
            is_required = check.name in required_names
            priority = "REQUIRED" if is_required else "recommended"
            print(f"  [{priority:11}] {check.fix}")

        if fix:
            # Auto-fix mode for instruction file
            print()
            print("Auto-fixing instruction file...")
            if file_path:
                try:
                    from kernle.cli.commands.init import generate_section, has_kernle_section

                    content = file_path.read_text()
                    if not has_kernle_section(content):
                        section = generate_section(
                            stack_id, style="combined", include_per_message=True
                        )
                        new_content = content.rstrip() + "\n\n" + section
                        file_path.write_text(new_content)
                        print(f"  ✓ Added Kernle instructions to {file_path}")
                    else:
                        print("  ⚠ File already has some Kernle instructions.")
                except Exception as e:
                    print(f"  ✗ Auto-fix failed: {e}")
            else:
                print("  ⚠ No instruction file to fix. Run `kernle init` first.")

            print()
            print("Note: --fix only updates the instruction file.")
            print("For beliefs: kernle migrate seed-beliefs")
            print("For hooks: kernle setup openclaw (or claude-code)")
    else:
        print()
        print("✓ All checks passed!")

    if not full_check:
        print()
        print("─" * 50)
        print("Run `kernle doctor --full` for comprehensive check")
        print("(includes seed beliefs and platform hooks)")
    else:
        print()
        print(f"Test: kernle -s {stack_id} load && kernle -s {stack_id} anxiety -b")


def cmd_doctor_structural(args, k: "Kernle"):
    """Run structural health check on memory graph.

    Privacy: outputs IDs and scores only, never memory content.
    """
    from kernle.structural import run_structural_checks

    output_json = getattr(args, "json", False)
    save_note = getattr(args, "save_note", False)

    findings = run_structural_checks(k)

    errors = [f for f in findings if f.severity == "error"]
    warnings = [f for f in findings if f.severity == "warning"]
    infos = [f for f in findings if f.severity == "info"]

    summary = {
        "total_findings": len(findings),
        "errors": len(errors),
        "warnings": len(warnings),
        "info": len(infos),
    }

    if output_json:
        output = {
            "summary": summary,
            "findings": [f.to_dict() for f in findings],
        }
        print(json.dumps(output, indent=2))
    else:
        print()
        print("=" * 55)
        print("  Kernle Doctor - Structural Health Check")
        print("=" * 55)
        print(f"  Stack: {k.stack_id}")
        print()

        if not findings:
            print("  All structural checks passed. Memory graph is healthy.")
            print()
            return

        print(f"  Findings: {len(errors)} errors, " f"{len(warnings)} warnings, {len(infos)} info")
        print()

        if errors:
            print("-" * 55)
            print("ERRORS")
            print("-" * 55)
            for f in errors:
                print(f"  [error]   {f.message}")
            print()

        if warnings:
            print("-" * 55)
            print("WARNINGS")
            print("-" * 55)
            for f in warnings:
                print(f"  [warning] {f.message}")
            print()

        if infos:
            print("-" * 55)
            print("INFO")
            print("-" * 55)
            for f in infos:
                print(f"  [info]    {f.message}")
            print()

    if save_note and findings:
        lines = [
            f"Structural health check: {summary['errors']}E / "
            f"{summary['warnings']}W / {summary['info']}I"
        ]
        for f in findings:
            lines.append(f"  [{f.severity}] {f.check}: {f.memory_type}:{f.memory_id[:12]}")
        note_content = "\n".join(lines)

        from kernle.types import Note as NoteModel

        note = NoteModel(
            id=str(uuid.uuid4()),
            stack_id=k.stack_id,
            content=note_content,
            note_type="diagnostic",
            reason="kernle doctor structural",
            created_at=datetime.now(timezone.utc),
            source_type="observation",
            source_entity="kernle:doctor",
        )
        k._storage.save_note(note)
        if not output_json:
            print(f"  Saved diagnostic note: {note.id[:12]}...")
            print()


def __getattr__(name):
    # Structural symbols -> deprecation warning, import from kernle.structural
    _moved_to_structural = {
        "StructuralFinding",
        "run_structural_checks",
        "check_orphaned_references",
        "check_low_confidence_beliefs",
        "check_stale_relationships",
        "check_belief_contradictions",
        "check_stale_goals",
    }
    if name in _moved_to_structural:
        import warnings

        warnings.warn(
            f"Importing {name} from kernle.cli.commands.doctor is deprecated. "
            f"Import from kernle.structural instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from kernle import structural

        return getattr(structural, name)

    # Session/report symbols -> clear error, NO reverse import from kernle_devtools
    _moved_to_devtools = {
        "cmd_doctor_session_start",
        "cmd_doctor_session_list",
        "cmd_doctor_report",
        "_check_operator_consent",
        "_generate_report_findings",
        "_recommendation_for",
        "_generate_summary",
    }
    if name in _moved_to_devtools:
        raise ImportError(
            f"{name} has moved to kernle_devtools.admin_health.diagnostics. "
            f"Install kernle-devtools: pip install kernle-devtools"
        )

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
