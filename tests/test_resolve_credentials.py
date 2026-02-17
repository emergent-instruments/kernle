"""Tests for kernle.credentials — shared credential resolution module."""

import json
import os
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Priority / fallback tests
# ---------------------------------------------------------------------------


class TestResolvePriority:
    """Credential resolution follows 3-tier priority:
    credentials.json > env vars > legacy config.json.
    """

    def test_resolve_from_credentials_json(self, tmp_path):
        """All fields returned when credentials.json has them."""
        creds = {
            "backend_url": "https://api.example.com",
            "auth_token": "tok123",
            "user_id": "user1",
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        with patch.dict(os.environ, {"KERNLE_DATA_DIR": str(tmp_path)}, clear=False):
            # Clear any env vars that would interfere
            env_clean = {
                k: v
                for k, v in os.environ.items()
                if k not in ("KERNLE_BACKEND_URL", "KERNLE_AUTH_TOKEN", "KERNLE_USER_ID")
            }
            env_clean["KERNLE_DATA_DIR"] = str(tmp_path)
            with patch.dict(os.environ, env_clean, clear=True):
                from kernle.credentials import resolve_credentials

                result = resolve_credentials()

        assert result["backend_url"] == "https://api.example.com"
        assert result["auth_token"] == "tok123"
        assert result["user_id"] == "user1"

    def test_resolve_from_env_vars(self, tmp_path):
        """All fields from env vars when no files exist."""
        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
            "KERNLE_USER_ID": "envuser",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "envtok"
        assert result["user_id"] == "envuser"

    def test_resolve_from_legacy_config(self, tmp_path):
        """Fields from legacy config.json when no credentials.json or env vars."""
        config = {
            "backend_url": "https://legacy.example.com",
            "auth_token": "legtok",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://legacy.example.com"
        assert result["auth_token"] == "legtok"

    def test_priority_credentials_json_over_env(self, tmp_path):
        """credentials.json wins when both file and env vars are set."""
        creds = {
            "backend_url": "https://file.example.com",
            "auth_token": "filetok",
            "user_id": "fileuser",
        }
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
            "KERNLE_USER_ID": "envuser",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://file.example.com"
        assert result["auth_token"] == "filetok"
        assert result["user_id"] == "fileuser"

    def test_priority_env_over_legacy(self, tmp_path):
        """Env vars win over legacy config.json."""
        config = {
            "backend_url": "https://legacy.example.com",
            "auth_token": "legtok",
        }
        (tmp_path / "config.json").write_text(json.dumps(config))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "envtok"

    def test_all_none_when_nothing_configured(self, tmp_path):
        """All fields None when no files or env vars exist."""
        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] is None
        assert result["auth_token"] is None
        assert result["user_id"] is None


# ---------------------------------------------------------------------------
# Token alias tests
# ---------------------------------------------------------------------------


class TestTokenAliases:
    """auth_token field supports aliases: auth_token > token > api_key."""

    def test_auth_token_field_takes_priority(self, tmp_path):
        """auth_token field used when present alongside token."""
        creds = {"auth_token": "preferred", "token": "fallback"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["auth_token"] == "preferred"

    def test_token_field_alias(self, tmp_path):
        """token field used as auth_token when auth_token is absent."""
        creds = {"token": "tok-via-alias"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["auth_token"] == "tok-via-alias"

    def test_api_key_field_alias(self, tmp_path):
        """api_key field used as auth_token when auth_token and token are absent."""
        creds = {"api_key": "key-via-alias"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["auth_token"] == "key-via-alias"

    def test_alias_precedence_auth_token_over_token_over_api_key(self, tmp_path):
        """When all three fields are present, auth_token wins."""
        creds = {"auth_token": "best", "token": "second", "api_key": "third"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["auth_token"] == "best"


# ---------------------------------------------------------------------------
# Per-field merge tests
# ---------------------------------------------------------------------------


class TestPerFieldMerge:
    """Each field is resolved independently across tiers."""

    def test_backend_url_from_file_token_from_env(self, tmp_path):
        """credentials.json has backend_url, env has auth_token."""
        creds = {"backend_url": "https://file.example.com"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_AUTH_TOKEN": "envtok",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://file.example.com"
        assert result["auth_token"] == "envtok"

    def test_backend_url_from_env_token_from_legacy(self, tmp_path):
        """Env has backend_url, legacy config.json has auth_token."""
        config = {"auth_token": "legtok"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "legtok"

    def test_user_id_from_file_rest_from_env(self, tmp_path):
        """File has user_id only, env has backend_url and auth_token."""
        creds = {"user_id": "fileuser"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "envtok"
        assert result["user_id"] == "fileuser"

    def test_all_fields_from_different_tiers(self, tmp_path):
        """Each field comes from a different tier."""
        # backend_url from credentials.json
        creds = {"backend_url": "https://file.example.com"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        # auth_token from env
        # user_id not in env; legacy has it... but legacy doesn't support user_id
        # So let's use: backend_url from file, auth_token from env, user_id from env
        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_AUTH_TOKEN": "envtok",
            "KERNLE_USER_ID": "envuser",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://file.example.com"
        assert result["auth_token"] == "envtok"
        assert result["user_id"] == "envuser"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: malformed files, empty strings, URL validation."""

    def test_malformed_credentials_json(self, tmp_path):
        """Invalid JSON in credentials.json falls through to env/legacy."""
        (tmp_path / "credentials.json").write_text("{not valid json!!!")

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "envtok"

    def test_partially_populated_credentials_json(self, tmp_path):
        """File has some fields, env/legacy consulted for rest."""
        creds = {"backend_url": "https://file.example.com"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        config = {"auth_token": "legtok"}
        (tmp_path / "config.json").write_text(json.dumps(config))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://file.example.com"
        assert result["auth_token"] == "legtok"

    def test_validate_backend_url_strips_trailing_slash(self, tmp_path):
        """Trailing slash is stripped from backend_url."""
        creds = {"backend_url": "https://api.example.com/"}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {"KERNLE_DATA_DIR": str(tmp_path)}
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://api.example.com"
        assert not result["backend_url"].endswith("/")

    def test_empty_strings_treated_as_none(self, tmp_path):
        """Empty string values in credentials.json treated as not set."""
        creds = {"backend_url": "", "auth_token": "", "user_id": ""}
        (tmp_path / "credentials.json").write_text(json.dumps(creds))

        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_BACKEND_URL": "https://env.example.com",
            "KERNLE_AUTH_TOKEN": "envtok",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] == "https://env.example.com"
        assert result["auth_token"] == "envtok"

    def test_env_vars_not_broadened(self, tmp_path):
        """Only the 3 specified env vars are checked (not arbitrary ones)."""
        env = {
            "KERNLE_DATA_DIR": str(tmp_path),
            "KERNLE_API_KEY": "should-not-be-used",
            "KERNLE_TOKEN": "should-not-be-used",
            "KERNLE_URL": "should-not-be-used",
        }
        with patch.dict(os.environ, env, clear=True):
            from kernle.credentials import resolve_credentials

            result = resolve_credentials()

        assert result["backend_url"] is None
        assert result["auth_token"] is None
        assert result["user_id"] is None
