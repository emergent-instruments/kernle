# Service: Local Development and Verification

## Purpose

Provide one clean, offline-first way to install, test, build, and smoke Kernle without relying on the current broken `.venv` or unavailable `uv`.

## Required Config and Secrets

- No secrets for deterministic tests.
- Set `KERNLE_DATA_DIR` to a new temporary directory for every test process. The runtime does **not** honor `KERNLE_HOME`.
- Also set `HOME` to a temporary directory for installed-command/E2E tests so an overlooked default cannot reach real user data.
- Use `KERNLE_STACK_ID` only when the test explicitly covers environment resolution.

## Human Setup Steps

Until Bundle 00 publishes canonical commands, a safe isolated baseline is:

```bash
export KERNLE_DATA_DIR="$(mktemp -d /tmp/kernle-data.XXXXXX)"
export HOME="$(mktemp -d /tmp/kernle-home.XXXXXX)"
export VENV_DIR="$(mktemp -d /tmp/kernle-venv.XXXXXX)"
python3.12 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -e '.[dev]'
"$VENV_DIR/bin/python" -m pytest tests/ -q
```

For package checks, additionally install `build` and `twine` in that environment. Do not reuse or delete the current broken `.venv` without owner approval.

## Allowed and Forbidden Behavior

- Allowed: temporary databases/HOMEs, local package builds, read-only GitHub checks.
- Forbidden: real-stack deletion, remote DB migration, provider calls from deterministic tests, release publishing, or cleanup of user-owned dirty files.
- Never run `supabase db reset`, `supabase db push`, or any remote equivalent; Supabase is not part of the recommended alpha.

## Test and Smoke Requirements

- Ruff and a pinned Black version.
- Targeted tests, full Python matrix, and 85% branch gate.
- Build plus Twine validation.
- Fresh base and each-supported-extra wheel import/entry-point smoke.
- CLI/MCP quickstart against temporary HOME.
- OpenClaw test/lint/build/package checks only if retained.
- Docs link/example validation.

## Current Blockers

- README requires `uv`, which is not installed.
- Current ignored `.venv` references a missing interpreter.
- Several existing tests set ignored `KERNLE_HOME` rather than `KERNLE_DATA_DIR` and can write into real `~/.kernle` data.
- Current Black check fails on `tests/test_mcp.py`; formatter versions are not bounded enough for stable CI.
- Base wheel exposes `kernle-mcp` without installing `mcp`.
- `uv.lock` is stale relative to `pyproject.toml`.
