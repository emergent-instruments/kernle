# Bundle: CLI and MCP Policy Boundaries

Type: service

Owns: minimal CLI init/status/diagnostics plus MCP request identity, retained tools, handlers, and output/error states. Installed-product release smoke belongs to Bundle 11.

Status: blocked on Bundle 08.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API/UI Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

An operator/agent can run the canonical workflows through CLI/MCP with explicit identity, actionable denial, no silent strict-mode mismatch, and no policy bypass.

## Services

- MCP runtime as a real installed dependency for the supported entry point.

## Libraries

- Existing CLI/parser and MCP package; align optional extras and console scripts honestly.

## API / Components

- Expose only the retained raw capture/promotion/get/search model.
- Include requester/owner/context/consent information in MCP contracts where required.
- Use stable machine-readable reason codes plus concise operator messages.

## Implementation Notes

- Replace tests that force `strict=False` when production is strict.
- Public examples must be executable, not illustrative pseudo-APIs.

## Unit Tests

- Schema validation, omitted requester, forged source, denial, malformed input, and unavailable model.

## Integration Tests

- Real CLI dispatcher and MCP handlers against temporary SQLite; no mock-only production path.

## E2E Tests

- MCP fixture flow for init/capture/promote/get/search/denial.
- MCP protocol fixture for the selected version.

## Raw Fixtures

- `tests/fixtures/mcp/<protocol-version>/` and product workflow/security fixtures.

## Done

- CLI and MCP cannot bypass SDK policy.
- Strict default and documented commands agree.
- Unsupported commands/tools are removed or explicitly deferred.
- MCP integration contracts pass; installed-wheel canary belongs to Bundle 11.
