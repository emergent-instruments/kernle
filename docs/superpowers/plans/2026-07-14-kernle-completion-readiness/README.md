# Kernle Completion Readiness Packet

This packet defines the smallest credible path from the current repository to a product that can be implemented by parallel coding agents without oral context.

**Verdict:** the repository is ready for stabilization and narrow integrity fixes, but **not** for an autonomous feature fleet. Product authority, public contracts, privacy/provenance rules, minimal workflow/security/MCP evidence, and the current docs worktree must be resolved first. Broader semantic/model qualification is post-alpha and claim-conditional.

## Canonical Packet Files

- [Readiness audit](readiness-audit.md): findings, evidence, verification, and the go/no-go decision.
- [Alpha simplification](alpha-simplification.md): recommended product cut and explicit deferrals.
- [Assignment context](assignment-context.md): required reading and claim rules for every agent.
- [Implementation plan](implementation-plan.md): dependency-ordered fleet map.
- [Owner inputs required](source-docs/inputs-required.md): exact decisions, fixtures, access, and target paths.
- [UI and operator guidance](ui-guidance.md): CLI, MCP, SDK, and integration states that must be specified.

Do not implement from either untracked roadmap alone. `dev-docs/ROADMAP.md` is the best current technical sequence, but it is proposed, overlaps other plans, and is not a product contract.

## Source Material Reviewed

See [source-docs/README.md](source-docs/README.md). The review covered repository authority, package and runtime code, Python and OpenClaw tests, CI, public docs, untracked internal roadmaps, local git state, and the live public GitHub backlog/workflow state.

## Recommended Alpha

A developer-operated, local SQLite memory runtime with:

- one canonical `Entity` + `Stack` Python API;
- MCP as the sole alpha host and CLI only for init/status/diagnostics;
- raw evidence plus one owner-approved structured type, recommended episode;
- explicit raw-to-structured promotion, get/search, and one export/restore path;
- owner-only request identity, per-stack SQLite, and immutable provenance;
- deterministic offline tests with no required provider.

OpenClaw, Claude Code host support, hosted cloud, sync, dashboards, additional memory types, model-driven processing, new adapters, pending/pondering, skills, cross-entity access/trust, and advanced cognition remain outside the first credible slice unless the owner explicitly replaces this cut.

## Build Order

1. [Repository baseline](bundles/00-repository-baseline.md) — owner confirmation required for the `audits/` to `dev-docs/` transition.
2. [Product contract freeze](bundles/01-product-contract-freeze.md) — owner-authored decisions required.
3. [Migration atomicity](bundles/03-migration-atomicity.md), the one unambiguous behavior repair after baseline.
4. After the relevant owner decisions:
   - [Stack isolation](bundles/04-stack-isolation.md).
   - [Atomic write integrity](bundles/02-atomic-write-integrity.md) only for retained affected types; otherwise close it as removed/not applicable.
5. [Security contract foundation](bundles/05-security-contract-foundation.md) after the minimal owner fixtures.
6. [API consolidation](bundles/06-api-consolidation.md).
7. [Policy and provenance core](bundles/07-policy-provenance-core.md).
8. [SDK and Stack policy boundaries](bundles/08-sdk-stack-policy-boundaries.md).
9. Parallel after the SDK/Stack boundary is green:
   - [CLI and MCP policy boundaries](bundles/09-cli-mcp-policy-boundaries.md).
   - [Serialization policy boundaries](bundles/10-serialization-policy-boundaries.md).
10. [Release and qualification gate](bundles/11-release-and-qualification.md).
11. [OpenClaw repair](bundles/99-openclaw-capture-integrity.md) only if Bundle 01 explicitly retains OpenClaw; otherwise it is non-gating deferred work.
12. Reassess other product features only after the retained alpha is green.

## Human-Intended Setup

- [Local development](services/local-development.md): clean environment and reproducible checks.
- [Local test-data cleanup](services/local-data-cleanup.md): read-only inventory first; deletion requires explicit approval.
- [Live model qualification](services/live-model-qualification.md): post-alpha only if a model-quality claim is approved.
- [Owner inputs required](source-docs/inputs-required.md): product choices and private-material handling.

## Samples and Fixtures

See [samples/README.md](samples/README.md). Existing tests provide broad deterministic regression coverage, but the qualification corpus is tiny and synthetic. Privacy/provenance truth tables, realistic multi-session cases, real host payloads, and supported-release migration snapshots are missing.

## Cross-Bundle References

- [Contract and schema ownership](schemas/README.md)
- [Fleet codebase guidelines](standards/codebase-guidelines.md)
- [Bundle audit protocol](audits/bundle-audit-protocol.md)
- [Local test-data inventory](audits/local-test-data-inventory.md)
- [SDK, CLI, MCP, and integration guidance](ui-guidance.md)

## Audit Feedback Incorporated

The packet incorporates independent product-readiness, architecture/security, test/CI, and packet-audit passes. In particular, it sequences around unsafe test-data isolation, the confirmed shared-database stack collision, unenforced privacy, forgeable/mutable provenance, strict-mode quickstart contradiction, OpenClaw failed-capture data loss, partial-migration versioning, atomic-update result loss, packaging/bootstrap drift, and lack of representative evaluation material.

## Bundle Types

- Worker/backend: atomic writes, policy/provenance core, SDK/Stack boundaries, serialization, and API consolidation.
- Schema: product contract plus stack isolation/migrations.
- Service: repository baseline, OpenClaw disposition, and CLI/MCP boundary.
- E2E: proof foundation.
- Workbench/frontend: none; the absent dashboard is deliberately deferred.

## Mockups and Current UI

See [mockups/README.md](mockups/README.md). The current product surfaces are SDK, CLI, MCP, and integrations; the documented dashboard has no implementation or design authority and is deferred.

## Fleet Pattern

- One coordinator owns dependency order and contract changes.
- One agent owns one bundle and one narrow file boundary.
- Contract authors work serially; boundary implementations may fan out only after contracts and fixtures merge.
- Every implementation bundle gets an independent audit using [the audit protocol](audits/bundle-audit-protocol.md).
- Agents may use public docs and GitHub read-only. They may not infer owner policy, use private memories as fixtures, mutate remote data, or publish a release without explicit authority.
