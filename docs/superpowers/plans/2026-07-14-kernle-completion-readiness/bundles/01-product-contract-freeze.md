# Bundle: Product Contract Freeze

Type: schema

Owns: alpha product contract, public-surface status, API/write/isolation decisions, platform support, and acceptance-scenario definitions. It does not implement runtime behavior.

Status: blocked on owner decisions in [inputs-required.md](../source-docs/inputs-required.md).

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema/API/UI Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Every contributor can answer who the alpha is for, what succeeds, what is unsafe, which surface is canonical, and what is explicitly deferred.

## Services

- None.

## Libraries

- No new dependencies.

## Schema / API / Components

- `docs/product/alpha-brief.md`
- decisions 0001-0006 from [inputs-required.md](../source-docs/inputs-required.md)
- `docs/product/surface-status.yaml`
- `docs/product/platform-support.md`
- `tests/fixtures/product/workflow-acceptance.yaml`

## Implementation Notes

- Source docs and observed behavior may inform choices but cannot make owner policy.
- Prefer one API, MCP only, owner-only local identity, one storage model, raw plus one structured type, and deliberate promotion.
- Mark each contradicted public claim implemented, partial, planned, deferred, or removed.

## Unit Tests

- Validate required decision/status fields and unique surface identifiers.

## Integration Tests

- Every implemented alpha surface names an existing test path.

## E2E Tests

- The three canonical workflows have observable acceptance steps and failure states.

## Raw Fixtures

- `tests/fixtures/product/workflow-acceptance.yaml` after owner approval.

## Done

- Owner approves the alpha brief and six decisions.
- Canonical API, host, trust root, retained types/lifecycle, write model, owner-only access, isolation, non-goals, and version meaning are unambiguous.
- Every public surface has one status and test owner.
- No two authoritative docs contradict the contract.
