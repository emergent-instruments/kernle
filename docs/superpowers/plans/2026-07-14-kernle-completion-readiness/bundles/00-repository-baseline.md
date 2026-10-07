# Bundle: Repository Baseline

Type: service

Owns: root authority, internal-docs tracking, safe test-data isolation, reproducible core development commands, formatter/tool pinning, and lock freshness. It does not choose or repair product-facing APIs, entry points, or quickstarts.

Status: blocked only on the owner's intended `audits/` to `dev-docs/` outcome and permission to modify existing dirty files.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

A fresh agent can identify the actual project, install core development dependencies, and run isolated targeted/full tests without relying on a stale local environment or touching real memory.

## Services

- [Local development](../services/local-development.md): clean bootstrap and smoke commands.

## Libraries

- Existing pip/pytest/ruff/black/build/twine toolchain.
- Choose pip/venv as the documented baseline unless the owner explicitly adds and locks `uv`.

## Implementation Notes

- Confirm whether `dev-docs/` replaces `audits/`, remains private, and keeps submodule history.
- Repair the move atomically; do not discard either checkout or user edits.
- Replace Supabase/Next/Expo authority with Python/SQLite/CLI/MCP/OpenClaw facts.
- Remove mechanical dead backend/web targets and workflow; record product-facing broken docs in surface status for Bundle 01 instead of deciding their replacement here.
- Refresh or remove stale `uv.lock` (it identifies Kernle 0.13.3 while `pyproject.toml` is 0.15.0).
- Defer MCP/other entry-point and public quickstart choices to Bundles 01, 09, and 11.
- Treat `.codex/config.toml` as non-authoritative until the selected entry point is installed and proven.
- Fix test fixtures that set ignored `KERNLE_HOME`; enforce a temporary `KERNLE_DATA_DIR` and `HOME` before any full suite.

## Unit Tests

- Authority/link validation rejects missing internal-authority paths.
- Isolation guard rejects any attempt to resolve the real user data directory during tests.

## Integration Tests

- Fresh isolated environment installs `.[dev]`, collects tests, and runs the targeted suite.
- A guard test proves the suite cannot open or modify the real `~/.kernle` path.
- Core editable import and targeted/full tests run with temporary `KERNLE_DATA_DIR` and `HOME`.

## E2E Tests

- Not required; product-facing E2E belongs to Bundle 11.

## Raw Fixtures

- No product fixtures required; use temporary directories and generated stacks.

## Done

- Git status contains no ambiguous deleted-submodule/untracked-replacement pair.
- `AGENTS.md` and `CLAUDE.md` describe the actual project and mutation policy.
- One documented bootstrap works on a clean checkout.
- Mechanical authority/bootstrap docs contain no missing path.
- Ruff, pinned Black, safe targeted/full tests, and core build pass.
- Product-facing entry-point/quickstart gaps are enumerated in Bundle 01's surface-status input, not silently fixed here.
