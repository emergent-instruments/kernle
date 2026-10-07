# Bundle: SDK and Stack Policy Boundaries

Type: worker/backend

Owns: retained canonical Python API and Stack create/get/search routing through the approved owner-only policy core. It does not own CLI/MCP or export/restore.

Status: blocked on Bundles 04, 06, and 07.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Python callers can access only authorized memory and cannot forge provenance or bypass policy through alternate Stack methods.

## Services

- [Local development](../services/local-development.md).

## Libraries

- Existing package contracts; no new framework.

## API / Components

- Thread `RequestContext` through retained create/get/search operations.
- Route all retained calls through one decision path.
- Remove or quarantine raw storage exposure from the public API.

## Implementation Notes

- Break signatures cleanly rather than preserve implicit `None` fallback.
- Validate at the public trust boundary; do not duplicate policy in each getter.

## Unit Tests

- Approved owner/omitted/forged/mutation cases against Entity and Stack.

## Integration Tests

- Real SQLite owner/denied create/get/search behavior.
- Persisted audit events and protected lineage round-trip without loss.

## E2E Tests

- Canonical Python workflow from installed wheel.

## Raw Fixtures

- Product workflow, privacy, and provenance fixtures.

## Done

- Every retained SDK/Stack path requires or derives an authenticated context according to contract.
- No alternate getter/search/write bypass exists.
- Public SDK workflow and denial tests pass.
