"""Shared credential resolution for Kernle sync and cloud operations.

Single source of truth for the 3-tier credential priority:
1. ~/.kernle/credentials.json (preferred)
2. Environment variables (KERNLE_BACKEND_URL, KERNLE_AUTH_TOKEN, KERNLE_USER_ID)
3. ~/.kernle/config.json (legacy fallback)
"""

import json
import logging
import os
from typing import Dict, Optional

from kernle.core.validation import validate_backend_url
from kernle.utils import get_kernle_home

logger = logging.getLogger(__name__)


def resolve_credentials() -> Dict[str, Optional[str]]:
    """Resolve sync credentials using 3-tier priority.

    Priority:
    1. ~/.kernle/credentials.json (preferred)
    2. Environment variables (KERNLE_BACKEND_URL, KERNLE_AUTH_TOKEN, KERNLE_USER_ID)
    3. ~/.kernle/config.json (legacy fallback)

    Each field is resolved independently -- a field present in tier 1 will
    not be overridden by a later tier, but missing fields will fall through.

    Token field aliasing in credentials.json: auth_token > token > api_key.
    Empty strings are treated as not set.

    Returns:
        Dict with keys: backend_url, auth_token, user_id (any may be None).
    """
    backend_url: Optional[str] = None
    auth_token: Optional[str] = None
    user_id: Optional[str] = None

    # --- Tier 1: credentials.json ---
    credentials_path = get_kernle_home() / "credentials.json"
    if credentials_path.exists():
        try:
            with open(credentials_path) as f:
                creds = json.load(f)
                backend_url = creds.get("backend_url") or None
                # Support multiple auth token field names
                auth_token = (
                    creds.get("auth_token") or creds.get("token") or creds.get("api_key")
                ) or None
                user_id = creds.get("user_id") or None
        except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
            logger.debug("Failed to load credentials file: %s", e, exc_info=True)

    # --- Tier 2: Environment variables ---
    if not backend_url:
        backend_url = os.environ.get("KERNLE_BACKEND_URL") or None
    if not auth_token:
        auth_token = os.environ.get("KERNLE_AUTH_TOKEN") or None
    if not user_id:
        user_id = os.environ.get("KERNLE_USER_ID") or None

    # --- Tier 3: Legacy config.json ---
    if not backend_url or not auth_token:
        config_path = get_kernle_home() / "config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
                    if not backend_url:
                        backend_url = config.get("backend_url") or None
                    if not auth_token:
                        auth_token = config.get("auth_token") or None
            except (json.JSONDecodeError, OSError, KeyError, ValueError) as e:
                logger.debug("Failed to load legacy config file: %s", e, exc_info=True)

    # --- Validation ---
    if backend_url:
        validated = validate_backend_url(backend_url)
        if validated:
            backend_url = validated.rstrip("/")
        else:
            backend_url = None

    return {
        "backend_url": backend_url,
        "auth_token": auth_token,
        "user_id": user_id,
    }
