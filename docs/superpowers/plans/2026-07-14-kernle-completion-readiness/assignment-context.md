# Assignment Context

Every agent must read this file and its bundle before changing code.

## Required Packet Context

1. [README](README.md)
2. [Readiness audit](readiness-audit.md)
3. [Alpha simplification](alpha-simplification.md)
4. The claimed bundle, its linked services, fixtures, schemas, standards, and tests
5. Current root `AGENTS.md` only after Bundle 00 corrects its false project context

Until Bundle 00 is complete, user instructions and this review packet outrank the copied Supabase/Next/Expo project section. Existing user-owned dirty files must be preserved.

## Extra Context for SDK, CLI, MCP, and Integration Bundles

- Trace the actual installed command/API path, not only unit helpers.
- Treat help, empty, error, denial, no-model, offline, timeout, and recovery states as contract behavior.
- Do not claim a host version without a real versioned payload fixture and canary.
- Do not add another public path when an existing one can be removed or delegated.

## Extra Context for Storage, Policy, and Migration Bundles

- Requester identity and trusted source identity cannot come from caller-controlled descriptive strings.
- Storage filtering is defense in depth, not the sole privacy boundary.
- Ordinary writes cannot mutate protected provenance.
- Migration success and schema-version advancement must be atomic.
- No remote database mutations are authorized.
- Never delete local data; create isolated fixtures/databases.

## Claim Rules

- One agent claims one bundle and records the exact files it owns.
- No implementation claim may cross a blocked dependency.
- Write a failing behavioral/contract test before changing non-trivial behavior.
- Run the smallest relevant test during work and the bundle verification at handoff.
- Report skipped tests, environment gaps, and residual risks explicitly.
- Do not commit, push, publish, migrate remotely, or change user-owned dirty files without direct authorization.
- A bundle is not done until an independent audit produces no unresolved P0/P1 finding.

