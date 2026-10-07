# Bundle: Release and Qualification Gate

Type: e2e

Owns: installed-wheel alpha workflows, executable docs, MCP host canary, surface/version/package parity, and only the additional qualification evidence for claims the owner retained. It does not add product behavior.

Status: blocked on Bundles 02-10. OpenClaw and live-model evidence are non-gating unless the owner explicitly adds those claims.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema/API/UI Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

A new developer can install the built artifact and complete the approved owner-only MCP and export/restore workflows exactly as documented.

## Services

- [Local development](../services/local-development.md).
- [Live model qualification](../services/live-model-qualification.md) only if model-driven behavior is added back to the claimed release.

## Libraries

- Existing build/Twine/pytest tooling and selected MCP dependency.
- Add only the docs/example checker needed to execute claimed snippets.

## Schema / API / Components

- Package version, lock, changelog, tag/release plan, extras, entry points, docs surface status, and supported Python/MCP versions agree.
- Base and each retained extra install independently without relying on transitive dependencies.
- Unsupported legacy commands/integrations are absent or clearly marked deferred.

## Implementation Notes

- Run all E2E under temporary `KERNLE_DATA_DIR` and `HOME`.
- A host fixture proves protocol compatibility; mocks alone are insufficient.
- Require migration snapshots only for upgrade origins named in Bundle 01.
- Require semantic/search/provider evidence only for claims reintroduced after the minimal alpha.

## Unit Tests

- Surface-status and fixture manifests validate.
- Package metadata/lock/version consistency check.

## Integration Tests

- Build/Twine; clean base/extra imports and console-script smoke.
- Ruff, pinned Black, full safe Python matrix, coverage, docs snippets, and optional Node checks only for retained Node scope.

## E2E Tests

- Three approved installed-wheel workflows.
- Real versioned MCP payload/canary.
- Per-stack collision/isolation and export/restore recovery.
- Privacy denial and provenance mutation denial.

## Raw Fixtures

- Minimal workflow/privacy/provenance fixtures from Bundle 05.
- `tests/fixtures/mcp/<protocol-version>/`.
- `tests/fixtures/migrations/<release>/` only for claimed upgrade origins.
- Qualification/search/provider corpora only for explicitly retained semantic/model claims.

## Done

- Installed artifacts, docs, and claimed surfaces agree.
- Base and retained extras/entry points work from clean environments.
- No test reaches real `~/.kernle` data.
- Required E2E and one real MCP canary pass.
- Every deferred claim is removed or clearly labelled; optional evidence does not block the minimal alpha.

