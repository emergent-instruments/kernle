# Codebase Guidelines for Fleet Work

## Actual Stack

- Python 3.10-3.12 package, SQLite reference storage, CLI, MCP.
- TypeScript/Vitest OpenClaw integration, deferred unless selected.
- No Next.js, Expo, mobile app, or in-repo backend currently exists.

## Change Rules

- Verify behavior, state, data integrity, and security.
- Prefer clean pre-alpha breaks and deletion over compatibility shims.
- One concern per module and one implementation path per public operation.
- Preserve unrelated/user-owned dirty worktree changes.
- No remote mutations, releases, commits, or pushes without direct authorization.

## Security and Data

- Authenticate requester/source at trust boundaries; descriptive strings are not identity.
- Fail closed according to the approved access decision.
- Preserve provenance; repair is separate, privileged, and audited.
- Use isolated temporary data in tests; never delete a real stack.
- Secrets stay out of repo/logs/fixtures.
- File writes containing memories must be private, atomic, and crash/concurrency safe.

## Dependencies

- Reuse existing/standard-library functionality when small and clear.
- Pin tools whose output gates CI.
- Every optional extra gets an import/entry-point smoke.
- Remove stale dependencies/extras instead of relying on transitive packages.

## Tests

- Tests must exercise the real contract, not a tautological mock.
- Deterministic unit/contract/integration/E2E tests are required for completion.
- Live provider/host checks are opt-in and separately labelled.
- Report skips and coverage gaps; global coverage alone is not proof.
- Every bundle runs its focused tests plus all affected contracts before audit.

## Documentation

- Executable examples over illustrative pseudo-code.
- One status per surface: implemented, partial, planned, deferred, removed.
- Update behavior docs in the same bundle as behavior.
- No public privacy, compatibility, provider, or performance claim without its named test/fixture.
