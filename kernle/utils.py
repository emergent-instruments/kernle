"""
Kernle utilities - shared helper functions.
"""

import hashlib
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional


def get_kernle_home() -> Path:
    """Get the Kernle data directory. Uses KERNLE_DATA_DIR env var if set, otherwise ~/.kernle."""
    custom = os.environ.get("KERNLE_DATA_DIR")
    if custom:
        return Path(custom)
    return Path.home() / ".kernle"


def _get_git_root() -> Optional[str]:
    """Get the root of the current git repository, if in one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # Git not available or command failed — non-critical
    return None


def generate_default_stack_id() -> str:
    """Generate a default stack ID based on machine + project path.

    Combines:
    1. Machine identifier (hostname)
    2. Project path (git root or cwd)

    Returns a stable ID like 'auto-a1b2c3d4' that:
    - Same machine + same directory = same stack (consistent)
    - Different machine or path = different stack (isolated)

    The user can always override with explicit -s <name> or KERNLE_STACK_ID env var.
    """
    # Get machine identifier
    machine = platform.node() or "unknown"

    # Get project path: prefer git root, fall back to cwd
    git_root = _get_git_root()
    project_path = git_root if git_root else os.getcwd()

    # Normalize the path for consistent hashing
    project_path = os.path.normpath(os.path.abspath(project_path))

    # Combine and hash
    identity_string = f"{machine}:{project_path}"
    hash_digest = hashlib.sha256(identity_string.encode()).hexdigest()

    return f"auto-{hash_digest[:8]}"


def resolve_stack_id(explicit_id: Optional[str] = None) -> str:
    """Resolve the stack ID with fallback chain.

    Resolution order:
    1. Explicit ID passed as argument (highest priority)
    2. KERNLE_STACK_ID environment variable
    3. Auto-generated from machine + project path

    Args:
        explicit_id: Explicitly provided stack ID (e.g., from -s flag)

    Returns:
        The resolved stack ID
    """
    # 1. Explicit ID takes priority
    if explicit_id and explicit_id != "default":
        return explicit_id

    # 2. Check environment variable
    env_id = os.environ.get("KERNLE_STACK_ID")
    if env_id:
        return env_id

    # 3. Generate from machine + project
    return generate_default_stack_id()
