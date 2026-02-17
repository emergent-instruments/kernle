"""Tests for validate_backend_url from kernle.core.validation.

Verifies that the canonical ``validate_backend_url`` prevents bearer tokens
from leaking over plaintext HTTP to non-local endpoints.

Note: sync.py previously re-exported this as ``_validate_backend_url``.
Since credential resolution was consolidated into ``kernle.credentials``,
the canonical import is used directly.
"""

from kernle.core.validation import validate_backend_url


class TestValidateBackendUrl:
    """URL validation prevents bearer tokens leaking over plaintext HTTP."""

    def test_https_url_passes(self):
        assert validate_backend_url("https://api.example.com") == "https://api.example.com"

    def test_http_localhost_passes(self):
        assert validate_backend_url("http://localhost:8000") == "http://localhost:8000"

    def test_http_127_0_0_1_passes(self):
        assert validate_backend_url("http://127.0.0.1:8000") == "http://127.0.0.1:8000"

    def test_http_localhost_no_port_passes(self):
        assert validate_backend_url("http://localhost") == "http://localhost"

    def test_http_remote_rejected(self):
        assert validate_backend_url("http://evil.example.com") is None

    def test_ftp_scheme_rejected(self):
        assert validate_backend_url("ftp://example.com") is None

    def test_empty_string_returns_none(self):
        assert validate_backend_url("") is None

    def test_none_returns_none(self):
        assert validate_backend_url(None) is None

    def test_https_with_port_passes(self):
        assert validate_backend_url("https://api.example.com:443") == "https://api.example.com:443"

    def test_http_remote_with_port_rejected(self):
        assert validate_backend_url("http://evil.example.com:8080") is None
