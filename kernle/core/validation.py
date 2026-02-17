"""Input validation methods for Kernle.

Provides both the ``ValidationMixin`` (instance methods used by
:class:`~kernle.core.Kernle`) and standalone validation helpers used
by CLI and MCP layers for consistent input sanitization.

Canonical helpers (used via import or re-export):
- ``sanitize_string`` — string validation + control-char stripping
- ``sanitize_number`` — numeric validation + NaN/Infinity rejection
- ``sanitize_list`` — array validation + null-item rejection
"""

import logging
import math
import re
from pathlib import Path
from typing import Any, List, Optional

logger = logging.getLogger(__name__)


def sanitize_string(
    value: Any, field_name: str, max_length: int = 1000, required: bool = True
) -> str:
    """Sanitize and validate string inputs.

    Canonical implementation shared by CLI (as ``validate_input``) and
    MCP (as ``sanitize_string``) layers.

    Args:
        value: The value to sanitize.
        field_name: Name of the field for error messages.
        max_length: Maximum allowed string length.
        required: If True, empty strings are rejected.

    Returns:
        Sanitized string.

    Raises:
        ValueError: If validation fails.
    """
    if value is None and not required:
        return ""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string, got {type(value).__name__}")

    if required and not value.strip():
        raise ValueError(f"{field_name} cannot be empty")

    if len(value) > max_length:
        raise ValueError(f"{field_name} too long (max {max_length} characters, got {len(value)})")

    # Remove null bytes and control characters except newlines and tabs
    sanitized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", value)

    return sanitized


def sanitize_number(
    value: Any,
    field_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    default: Optional[float] = None,
) -> float:
    """Validate numeric inputs, rejecting NaN and Infinity.

    Canonical implementation shared by MCP layer (as ``validate_number``).

    Args:
        value: The value to validate.
        field_name: Name of the field for error messages.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        default: Default value if *value* is None.

    Returns:
        Validated float.

    Raises:
        ValueError: If validation fails.
    """
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"{field_name} is required")

    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number, got bool")

    if not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {type(value).__name__}")

    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        raise ValueError(f"{field_name} must be a finite number, got {value}")

    if min_val is not None and value < min_val:
        raise ValueError(f"{field_name} must be >= {min_val}, got {value}")

    if max_val is not None and value > max_val:
        raise ValueError(f"{field_name} must be <= {max_val}, got {value}")

    return float(value)


def sanitize_list(
    value: Any,
    field_name: str,
    item_max_length: int = 500,
    max_items: int = 100,
) -> List[str]:
    """Validate and sanitize list/array inputs.

    Canonical implementation shared by MCP layer (as ``sanitize_array``).

    Args:
        value: The array to validate.
        field_name: Name of the field for error messages.
        item_max_length: Maximum length for each string item.
        max_items: Maximum number of items allowed.

    Returns:
        List of sanitized strings (empty items removed).

    Raises:
        ValueError: If validation fails.
    """
    if value is None:
        return []

    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be an array, got {type(value).__name__}")

    if len(value) > max_items:
        raise ValueError(f"{field_name} too many items (max {max_items}, got {len(value)})")

    if any(item is None for item in value):
        raise ValueError(f"{field_name} must not contain null items")

    sanitized = []
    for i, item in enumerate(value):
        sanitized_item = sanitize_string(
            item, f"{field_name}[{i}]", item_max_length, required=False
        )
        if sanitized_item:
            sanitized.append(sanitized_item)

    return sanitized


def validate_backend_url(url: str, *, allow_localhost_http: bool = True) -> "str | None":
    """Validate a backend URL for safe credential transmission.

    Canonical implementation shared by CLI sync and cloud storage layers.
    Rejects non-http/https schemes, URLs with no host, and remote HTTP
    endpoints (only localhost/127.0.0.1 are allowed over plaintext HTTP).

    Args:
        url: The backend URL to validate.
        allow_localhost_http: If True (default), permit ``http://localhost``
            and ``http://127.0.0.1``.  When False, *all* HTTP URLs are
            rejected regardless of host.

    Returns:
        The URL unchanged if valid, or ``None`` if rejected (with a
        warning logged for each rejection reason).
    """
    if not url:
        return None
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"https", "http"}:
        logger.warning("Invalid backend_url scheme; only http/https allowed.")
        return None
    if not parsed.netloc:
        logger.warning("Invalid backend_url; missing host.")
        return None
    if parsed.scheme == "http":
        if not allow_localhost_http:
            logger.warning("HTTP not allowed in this context.")
            return None
        host = parsed.hostname or ""
        if host not in {"localhost", "127.0.0.1"}:
            logger.warning("Refusing non-local http backend_url for security.")
            return None
    return url


class ValidationMixin:
    """Input validation operations for Kernle."""

    def _validate_stack_id(self, stack_id: str) -> str:
        """Validate and sanitize agent ID.

        Rejects path traversal attempts before sanitizing.
        """
        if not stack_id or not stack_id.strip():
            raise ValueError("Stack ID cannot be empty")

        stripped = stack_id.strip()

        # Reject path traversal characters and patterns
        if "/" in stripped or "\\" in stripped:
            raise ValueError("Stack ID must not contain path separators")
        if stripped == "." or stripped == "..":
            raise ValueError("Stack ID must not be a relative path component")
        if ".." in stripped.split("."):
            raise ValueError("Stack ID must not contain path traversal sequences")

        # Remove potentially dangerous characters
        sanitized = "".join(c for c in stripped if c.isalnum() or c in "-_.")

        if not sanitized:
            raise ValueError("Stack ID must contain alphanumeric characters")

        if len(sanitized) > 100:
            raise ValueError("Stack ID too long (max 100 characters)")

        return sanitized

    def _validate_checkpoint_dir(self, checkpoint_dir: Path) -> Path:
        """Validate checkpoint directory path."""
        import tempfile

        try:
            # Resolve to absolute path to prevent directory traversal
            resolved_path = checkpoint_dir.resolve()

            # Ensure it's within a safe directory (user's home, system temp, or /tmp)
            home_path = Path.home().resolve()
            tmp_path = Path("/tmp").resolve()
            system_temp = Path(tempfile.gettempdir()).resolve()

            # Use is_relative_to() for secure path validation (Python 3.9+)
            # This properly handles edge cases like /home/user/../etc that startswith() misses
            is_safe = (
                resolved_path.is_relative_to(home_path)
                or resolved_path.is_relative_to(tmp_path)
                or resolved_path.is_relative_to(system_temp)
            )

            # Also allow /var/folders on macOS (where tempfile creates dirs)
            if not is_safe:
                try:
                    var_folders = Path("/var/folders").resolve()
                    private_var_folders = Path("/private/var/folders").resolve()
                    is_safe = resolved_path.is_relative_to(
                        var_folders
                    ) or resolved_path.is_relative_to(private_var_folders)
                except (OSError, ValueError):
                    pass  # macOS realpath resolution failed — skip check

            if not is_safe:
                raise ValueError("Checkpoint directory must be within user home or temp directory")

            return resolved_path

        except (OSError, ValueError) as e:
            logger.error(f"Invalid checkpoint directory: {e}", exc_info=True)
            raise ValueError(f"Invalid checkpoint directory: {e}")

    def _validate_string_input(
        self, value: str, field_name: str, max_length: Optional[int] = 1000
    ) -> str:
        """Validate and sanitize string inputs.

        Args:
            value: String to validate
            field_name: Name of the field (for error messages)
            max_length: Maximum length, or None to skip length check

        Returns:
            Sanitized string
        """
        if not isinstance(value, str):
            raise TypeError(f"{field_name} must be a string")

        if max_length is not None and len(value) > max_length:
            raise ValueError(f"{field_name} too long (max {max_length} characters)")

        # Basic sanitization - remove null bytes and control characters
        sanitized = value.replace("\x00", "").replace("\r\n", "\n")

        return sanitized

    @staticmethod
    def _validate_derived_from(refs: Optional[List[str]]) -> Optional[List[str]]:
        """Validate derived_from references format.

        Accepts refs in format 'type:id' or 'context:description'.
        Filters out empty strings and validates basic structure.

        Args:
            refs: List of memory references

        Returns:
            Validated list, or None if empty
        """
        if not refs:
            return None

        valid_types = {
            "raw",
            "episode",
            "belief",
            "note",
            "value",
            "goal",
            "drive",
            "relationship",
            "context",
            "kernle",
        }
        validated = []
        for ref in refs:
            if not ref or not isinstance(ref, str):
                continue
            ref = ref.strip()
            if ":" not in ref:
                continue  # Skip malformed refs
            ref_type = ref.split(":", 1)[0]
            if ref_type not in valid_types:
                continue  # Skip unknown types
            validated.append(ref)

        return validated if validated else None
