"""Authentication commands for Kernle CLI."""

import json
import logging
import sys
from typing import TYPE_CHECKING, Any, Dict, Optional

from kernle.cli.commands.credentials import (
    clear_credentials,
    get_credentials_path,
    load_credentials,
    prompt_backend_url,
    require_https_url,
    save_credentials,
)

if TYPE_CHECKING:
    from kernle import Kernle

logger = logging.getLogger(__name__)


def _mask_secret(secret: str, prefix: int = 4, suffix: int = 4) -> str:
    """Mask a secret for safe display (never returns the full secret)."""
    if not secret:
        return ""

    if len(secret) <= prefix + suffix:
        if len(secret) <= 2:
            return "*" * len(secret)
        visible = max(1, len(secret) // 3)
        return f"{secret[:visible]}...{secret[-visible:]}"

    return f"{secret[:prefix]}...{secret[-suffix:]}"


def _sanitize_key_json_result(
    result: Dict[str, Any],
    *,
    fallback_id: Optional[str] = None,
    old_key_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Sanitize key create/cycle JSON output with a strict allowlist."""
    if not isinstance(result, dict):
        return {}

    sanitized: Dict[str, Any] = {}

    key_id = result.get("id") or result.get("key_id") or fallback_id
    if isinstance(key_id, str) and key_id:
        sanitized["id"] = key_id

    name = result.get("name")
    if isinstance(name, str):
        sanitized["name"] = name

    # The generated key must be returned once; normalize key/api_key to one field.
    raw_key = result.get("key") or result.get("api_key")
    if isinstance(raw_key, str) and raw_key:
        sanitized["key"] = raw_key

    key_prefix = result.get("key_prefix")
    if isinstance(key_prefix, str):
        sanitized["key_prefix"] = key_prefix

    created_at = result.get("created_at")
    if isinstance(created_at, str):
        sanitized["created_at"] = created_at

    expires_at = result.get("expires_at")
    if isinstance(expires_at, str):
        sanitized["expires_at"] = expires_at

    is_active = result.get("is_active")
    if isinstance(is_active, bool):
        sanitized["is_active"] = is_active

    if isinstance(old_key_id, str) and old_key_id:
        sanitized["old_key_id"] = old_key_id

    return sanitized


def cmd_auth(args, k: "Kernle" = None):
    """Handle auth subcommands."""
    from datetime import datetime, timezone

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    if args.auth_action == "register":
        httpx = get_http_client()

        # Load existing credentials to get backend_url if set
        existing = load_credentials()
        backend_url = args.backend_url or (existing.get("backend_url") if existing else None)

        # Prompt for backend URL if not provided
        if not backend_url:
            backend_url = prompt_backend_url()

        backend_url = backend_url.rstrip("/")

        # SECURITY: Warn about non-HTTPS URLs (credentials sent in cleartext)
        url_source = "args" if args.backend_url else ("credentials" if existing else None)
        require_https_url(backend_url, url_source)

        print(f"Registering with {backend_url}...")
        print()

        # Prompt for email if not provided
        email = args.email
        if not email:
            print("Email: ", end="", flush=True)
            try:
                email = input().strip()
                if not email:
                    print("✗ Email is required")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)

        # Get stack_id from Kernle instance
        stack_id = k.stack_id if k else "default"

        # Call registration endpoint
        try:
            response = httpx.post(
                f"{backend_url}/auth/register",
                json={"stack_id": stack_id, "email": email},
                timeout=30.0,
            )

            if response.status_code == 200 or response.status_code == 201:
                result = response.json()
                user_id = result.get("user_id")
                # Backend returns "secret" for the permanent credential (store as api_key)
                secret = result.get("secret")
                # Backend returns "access_token" for the JWT (store as auth_token)
                access_token = result.get("access_token")
                expires_in = result.get("expires_in", 604800)

                if not user_id or not secret:
                    print("✗ Registration failed: Invalid response from server")
                    print(f"  Response: {response.text[:200]}")
                    sys.exit(1)

                # Calculate token expiry
                token_expires = (
                    datetime.now(timezone.utc).isoformat()
                    if not expires_in
                    else (
                        datetime.now(timezone.utc)
                        + __import__("datetime").timedelta(seconds=expires_in)
                    ).isoformat()
                )

                # Save credentials
                # Note: Store secret as api_key for consistency with other auth flows
                credentials = {
                    "user_id": user_id,
                    "api_key": secret,  # Permanent secret stored as api_key
                    "backend_url": backend_url,
                    "auth_token": access_token,  # JWT access token (matches login flow)
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "success",
                                "user_id": user_id,
                                "backend_url": backend_url,
                            },
                            indent=2,
                        )
                    )
                else:
                    print("✓ Registration successful!")
                    print()
                    print(f"  User ID:     {user_id}")
                    print(f"  Stack ID:    {stack_id}")
                    print(
                        f"  Secret:      {_mask_secret(secret)}"
                    )  # masked: shows only first/last 4 chars
                    print(f"  Backend:     {backend_url}")
                    print()
                    print(f"Credentials saved to {get_credentials_path()}")

            elif response.status_code == 409:
                print("✗ Email already registered")
                print("  Use `kernle auth login` to log in with existing credentials")
                sys.exit(1)
            elif response.status_code == 400:
                error = response.json().get("detail", response.text)
                print(f"✗ Registration failed: {error}")
                sys.exit(1)
            else:
                print(f"✗ Registration failed: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            print("  Check that the backend URL is correct and the server is running")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Registration failed: {e}")
            sys.exit(1)

    elif args.auth_action == "login":
        httpx = get_http_client()

        # Load existing credentials
        existing = load_credentials()
        backend_url = args.backend_url or (existing.get("backend_url") if existing else None)
        api_key = args.api_key or (existing.get("api_key") if existing else None)

        # Prompt for backend URL if not provided
        if not backend_url:
            backend_url = prompt_backend_url()

        backend_url = backend_url.rstrip("/")

        # SECURITY: Warn about non-HTTPS URLs (credentials sent in cleartext)
        url_source = "args" if args.backend_url else ("credentials" if existing else None)
        require_https_url(backend_url, url_source)

        # Prompt for API key if not provided
        if not api_key:
            import getpass

            try:
                # SECURITY: Use getpass to hide input (prevents shoulder-surfing/log capture)
                api_key = getpass.getpass("API Key: ").strip()
                if not api_key:
                    print("✗ API Key is required")
                    sys.exit(1)
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                sys.exit(1)

        print(f"Logging in to {backend_url}...")

        # Call login endpoint to refresh token
        try:
            response = httpx.post(
                f"{backend_url}/auth/login",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 200:
                result = response.json()
                user_id = result.get("user_id")
                token = result.get("token")
                token_expires = result.get("token_expires")

                # Update credentials
                # Note: Use "auth_token" to match what storage layer expects
                credentials = {
                    "user_id": user_id,
                    "api_key": api_key,
                    "backend_url": backend_url,
                    "auth_token": token,  # Storage expects "auth_token", not "token"
                    "token_expires": token_expires,
                }
                save_credentials(credentials)

                if args.json:
                    print(
                        json.dumps(
                            {
                                "status": "success",
                                "user_id": user_id,
                                "token_expires": token_expires,
                            },
                            indent=2,
                        )
                    )
                else:
                    print("✓ Login successful!")
                    print()
                    print(f"  User ID:       {user_id}")
                    print(f"  Backend:       {backend_url}")
                    if token_expires:
                        print(f"  Token expires: {token_expires}")
                    print()
                    print(f"Credentials saved to {get_credentials_path()}")

            elif response.status_code == 401:
                print("✗ Invalid API key")
                sys.exit(1)
            else:
                print(f"✗ Login failed: HTTP {response.status_code}")
                print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            print("  Check that the backend URL is correct and the server is running")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Login failed: {e}")
            sys.exit(1)

    elif args.auth_action == "status":
        credentials = load_credentials()

        if not credentials:
            if args.json:
                print(
                    json.dumps({"authenticated": False, "reason": "No credentials found"}, indent=2)
                )
            else:
                print("Not authenticated")
                print()
                print("Run `kernle auth register` to create an account")
                print("Run `kernle auth login` to log in with an existing API key")
            return

        user_id = credentials.get("user_id")
        api_key = credentials.get("api_key")
        backend_url = credentials.get("backend_url")
        # Support both "auth_token" (preferred) and "token" (legacy) for backwards compatibility
        token = credentials.get("auth_token") or credentials.get("token")
        token_expires = credentials.get("token_expires")

        # Check if token is expired
        token_valid = False
        expires_in = None
        if token_expires:
            try:
                # Parse ISO format timestamp
                expires_dt = datetime.fromisoformat(token_expires.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                if expires_dt > now:
                    token_valid = True
                    delta = expires_dt - now
                    if delta.total_seconds() < 3600:
                        expires_in = f"{int(delta.total_seconds() / 60)} minutes"
                    elif delta.total_seconds() < 86400:
                        expires_in = f"{int(delta.total_seconds() / 3600)} hours"
                    else:
                        expires_in = f"{int(delta.total_seconds() / 86400)} days"
            except (ValueError, TypeError):
                pass  # Time formatting failed — omit expiry display

        if args.json:
            print(
                json.dumps(
                    {
                        "authenticated": True,
                        "user_id": user_id,
                        "backend_url": backend_url,
                        "has_api_key": bool(api_key),
                        "has_token": bool(token),
                        "token_valid": token_valid,
                        "token_expires": token_expires,
                    },
                    indent=2,
                )
            )
        else:
            print("Auth Status")
            print("=" * 40)
            print()

            auth_icon = "🟢" if token_valid else ("🟡" if api_key else "🔴")
            print(f"{auth_icon} Authenticated: {'Yes' if api_key else 'No'}")
            print()

            if user_id:
                print(f"  User ID:     {user_id}")
            if backend_url:
                print(f"  Backend:     {backend_url}")
            if api_key:
                print(
                    f"  API Key:     {_mask_secret(api_key)}"
                )  # masked: shows only first/last 4 chars

            if token:
                if token_valid:
                    print(f"  Token:       ✓ Valid (expires in {expires_in})")
                else:
                    print("  Token:       ✗ Expired")
                    print()
                    print("Run `kernle auth login` to refresh your token")
            else:
                print("  Token:       Not set")

            print()
            print(f"Credentials: {get_credentials_path()}")

    elif args.auth_action == "logout":
        creds_path = get_credentials_path()
        if clear_credentials():
            if args.json:
                print(json.dumps({"status": "success", "message": "Credentials cleared"}, indent=2))
            else:
                print("✓ Logged out")
                print(f"  Removed {creds_path}")
        else:
            if args.json:
                print(
                    json.dumps(
                        {"status": "success", "message": "No credentials to clear"}, indent=2
                    )
                )
            else:
                print("Already logged out (no credentials found)")

    elif args.auth_action == "keys":
        cmd_auth_keys(args)


def cmd_auth_keys(args):
    """Handle API key management subcommands."""

    def get_http_client():
        """Get an HTTP client for backend requests."""
        try:
            import httpx

            return httpx
        except ImportError:
            print("✗ httpx not installed. Run: pip install httpx")
            sys.exit(1)

    def require_auth():
        """Load credentials and return (backend_url, api_key) or exit."""
        credentials = load_credentials()
        if not credentials:
            print("✗ Not authenticated")
            print("  Run `kernle auth login` first")
            sys.exit(1)

        backend_url = credentials.get("backend_url")
        api_key = credentials.get("api_key")

        if not backend_url or not api_key:
            print("✗ Missing credentials")
            print("  Run `kernle auth login` to re-authenticate")
            sys.exit(1)

        require_https_url(backend_url, source="credentials")

        return backend_url.rstrip("/"), api_key

    def mask_key(key: str) -> str:
        """Mask an API key for display (show first 8 and last 4 chars)."""
        if not key or len(key) <= 16:
            return key[:4] + "..." if key and len(key) > 4 else key or ""
        return key[:8] + "..." + key[-4:]

    httpx = get_http_client()

    if args.keys_action == "list":
        backend_url, api_key = require_auth()

        try:
            response = httpx.get(
                f"{backend_url}/auth/keys",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code == 200:
                keys = response.json()

                if args.json:
                    print(json.dumps(keys, indent=2))
                else:
                    if not keys:
                        print("No API keys found.")
                        return

                    print("API Keys")
                    print("=" * 70)
                    print()

                    for key_info in keys:
                        key_id = key_info.get("id", "unknown")
                        name = key_info.get("name") or "(unnamed)"
                        masked = mask_key(key_info.get("key_prefix", ""))
                        created = (
                            key_info.get("created_at", "")[:10]
                            if key_info.get("created_at")
                            else "unknown"
                        )
                        last_used = key_info.get("last_used_at")
                        is_active = key_info.get("is_active", True)

                        status_icon = "🟢" if is_active else "🔴"
                        print(f"{status_icon} {name}")
                        print(f"   ID:       {key_id}")
                        print(f"   Key:      {masked}...")
                        print(f"   Created:  {created}")
                        if last_used:
                            print(f"   Last used: {last_used[:10]}")
                        if not is_active:
                            print("   Status:   REVOKED")
                        print()

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            else:
                print(f"✗ Failed to list keys: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception as e:
                    logger.debug(f"Failed to parse error response as JSON: {e}", exc_info=True)
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to list keys: {e}")
            sys.exit(1)

    elif args.keys_action == "create":
        backend_url, api_key = require_auth()
        name = getattr(args, "name", None)

        try:
            payload = {}
            if name:
                payload["name"] = name

            response = httpx.post(
                f"{backend_url}/auth/keys",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=30.0,
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_key = result.get("key") or result.get("api_key")
                key_id = result.get("id") or result.get("key_id")
                key_name = result.get("name") or name or "(unnamed)"

                if args.json:
                    print(
                        json.dumps(_sanitize_key_json_result(result, fallback_id=key_id), indent=2)
                    )
                else:
                    print("✓ API key created")
                    print()
                    print("=" * 70)
                    print("⚠️  SAVE THIS KEY NOW - IT WILL ONLY BE SHOWN ONCE!")
                    print("=" * 70)
                    print()
                    print(f"  Name:    {key_name}")
                    print(f"  ID:      {key_id}")
                    print(f"  Key:     {new_key}")
                    print()
                    print("=" * 70)
                    print()
                    print("Store this key securely. You will not be able to see it again.")
                    print("Use `kernle auth keys list` to see your keys (masked).")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 429:
                print("✗ Rate limit exceeded")
                print("  Wait a moment and try again")
                sys.exit(1)
            else:
                print(f"✗ Failed to create key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception as e:
                    logger.debug(f"Failed to parse error response as JSON: {e}", exc_info=True)
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to create key: {e}")
            sys.exit(1)

    elif args.keys_action == "revoke":
        backend_url, api_key = require_auth()
        key_id = args.key_id

        # Confirm unless --force
        if not getattr(args, "force", False):
            print(f"⚠️  You are about to revoke API key: {key_id}")
            print("   This action cannot be undone.")
            print()
            print("Type 'yes' to confirm: ", end="", flush=True)
            try:
                confirm = input().strip().lower()
                if confirm != "yes":
                    print("Aborted.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

        try:
            response = httpx.delete(
                f"{backend_url}/auth/keys/{key_id}",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code in (200, 204):
                if args.json:
                    print(
                        json.dumps(
                            {"status": "success", "key_id": key_id, "action": "revoked"}, indent=2
                        )
                    )
                else:
                    print(f"✓ API key {key_id} has been revoked")
                    print()
                    print("  The key can no longer be used for authentication.")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 404:
                print(f"✗ Key not found: {key_id}")
                sys.exit(1)
            else:
                print(f"✗ Failed to revoke key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception as e:
                    logger.debug(f"Failed to parse error response as JSON: {e}", exc_info=True)
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to revoke key: {e}")
            sys.exit(1)

    elif args.keys_action == "cycle":
        backend_url, api_key = require_auth()
        key_id = args.key_id

        # Confirm unless --force
        if not getattr(args, "force", False):
            print(f"⚠️  You are about to cycle API key: {key_id}")
            print("   The old key will be deactivated and a new key will be generated.")
            print()
            print("Type 'yes' to confirm: ", end="", flush=True)
            try:
                confirm = input().strip().lower()
                if confirm != "yes":
                    print("Aborted.")
                    return
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return

        try:
            response = httpx.post(
                f"{backend_url}/auth/keys/{key_id}/cycle",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )

            if response.status_code in (200, 201):
                result = response.json()
                new_key = result.get("key") or result.get("api_key")
                new_key_id = result.get("id") or result.get("key_id")
                key_name = result.get("name") or "(unnamed)"

                if args.json:
                    print(
                        json.dumps(
                            _sanitize_key_json_result(
                                result,
                                fallback_id=new_key_id,
                                old_key_id=key_id,
                            ),
                            indent=2,
                        )
                    )
                else:
                    print("✓ API key cycled")
                    print()
                    print(f"  Old key {key_id} has been deactivated.")
                    print()
                    print("=" * 70)
                    print("⚠️  SAVE THIS NEW KEY NOW - IT WILL ONLY BE SHOWN ONCE!")
                    print("=" * 70)
                    print()
                    print(f"  Name:    {key_name}")
                    print(f"  ID:      {new_key_id}")
                    print(f"  Key:     {new_key}")
                    print()
                    print("=" * 70)
                    print()
                    print("Update any systems using the old key to use this new key.")

            elif response.status_code == 401:
                print("✗ Authentication failed")
                print("  Run `kernle auth login` to re-authenticate")
                sys.exit(1)
            elif response.status_code == 404:
                print(f"✗ Key not found: {key_id}")
                sys.exit(1)
            else:
                print(f"✗ Failed to cycle key: HTTP {response.status_code}")
                try:
                    error = response.json().get("detail", response.text)
                    print(f"  {error[:200]}")
                except Exception as e:
                    logger.debug(f"Failed to parse error response as JSON: {e}", exc_info=True)
                    print(f"  {response.text[:200]}")
                sys.exit(1)

        except httpx.ConnectError:
            print(f"✗ Could not connect to {backend_url}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to cycle key: {e}")
            sys.exit(1)
