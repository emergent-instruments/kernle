# Bundle: Migration Atomicity

Type: schema

Owns: fail-fast schema migration execution, postcondition validation, transaction boundaries, and schema-version advancement. It does not choose the stack-isolation model or promise upgrade compatibility.

Status: claimable after Bundle 00; no owner policy is needed to stop partial migrations being marked current.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

A failed or incomplete migration leaves the database at its prior valid version and reports an actionable error.

## Services

- [Local development](../services/local-development.md); generated temporary databases only.

## Libraries

- SQLite and existing migration helpers. Do not add a framework unless one atomic local implementation cannot be expressed clearly.

## Schema

- Run a migration and its postconditions in one transaction where SQLite permits.
- Advance `schema_version` only after every required table/column/index/invariant succeeds.
- Fail loudly; warning-and-continue is not migration success.

## Implementation Notes

- TDD an injected mid-migration failure before changing the runner.
- Do not build historical upgrade paths in this bundle.

## Unit Tests

- Failure rolls back data and version.
- Missing postcondition rejects advancement.
- Re-running a successful migration is idempotent.

## Integration Tests

- Generated prior schema migrates atomically and reopens cleanly.

## E2E Tests

- Not required until an upgrade origin is claimed in Bundle 11.

## Raw Fixtures

- Generated minimal schemas; no owner material required.

## Done

- No caught migration error can lead to a current schema version.
- Postconditions cover each executed migration.
- Focused schema and migration tests pass under a temporary data directory.

