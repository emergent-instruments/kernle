# Bundle: Stack Isolation

Type: schema

Owns: the chosen SQLite stack path/key model and cross-stack collision prevention. Sync is deferred; upgrade compatibility belongs to Bundle 11 only if claimed.

Status: blocked on Bundle 03 and decision 0004. Recommended decision: one database file per stack and a clean pre-alpha reset path.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Two stacks cannot overwrite, search, load, or queue each other's records.

## Services

- [Local development](../services/local-development.md); isolated temporary databases only.

## Libraries

- SQLite and existing path/validation helpers; no new dependency.

## Schema

- Recommended alpha: one database file per stack, with explicit path naming, move, and clean-reset semantics. Bundle 10 exclusively owns data export/restore.
- Alternative only if owner-required: composite `(stack_id, id)` keys plus stack scope on every auxiliary/search/sync table and query. Do not retain both models.
- With per-stack files, disable/defer sync rather than redesign its shared auxiliary tables in alpha.

## Implementation Notes

- Never touch a real local stack; set temporary `KERNLE_DATA_DIR` and `HOME`.
- Remove `INSERT OR REPLACE` behavior that can delete another logical record or provenance unexpectedly.
- Define a clean reset marker/error path instead of implementing serialization or speculative upgrade compatibility here.

## Unit Tests

- Stack ID resolves to a stable, validated, non-colliding path.
- Path traversal and normalized-ID collisions fail.

## Integration Tests

- Two stacks can use the same record ID without collision.
- Search/load and any retained queue state remain stack-isolated.

## E2E Tests

- A temporary two-stack workflow restarts and recovers each stack independently.

## Raw Fixtures

- Generated two-stack collision fixtures; no owner data required.

## Done

- Chosen isolation model is implemented once and documented.
- Collision and cross-queue tests fail on old code and pass on new code.
- Real `~/.kernle` is never touched by tests.
- Breaking reset behavior is explicit; no serialization or unclaimed migration work is added.
