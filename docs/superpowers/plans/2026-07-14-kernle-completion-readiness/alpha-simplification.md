# Alpha Simplification

## Purpose

Cut the product to the smallest slice that can make a defensible safety and usefulness claim.

## First-Slice Goal

A developer can install Kernle, attach one owner-only local SQLite stack through one canonical `Entity` + `Stack` API and MCP, capture raw evidence, deliberately promote it to one structured memory type, retrieve it, and export/restore it with an auditable provenance chain.

## Non-Negotiables

- Owner-only private-by-default is enforced at every retained read/search/export boundary.
- Structured memory has authenticated, immutable origin and append-only lineage.
- A failed interception never destroys the source write.
- One operation has one canonical implementation path.
- Offline deterministic behavior does not require a model or network.
- Docs examples execute against the installed wheel.
- Breaking cleanup is preferred to compatibility shims while pre-alpha.

## Alpha Build Shape

- Python 3.10-3.12 package.
- SQLite reference storage only.
- `Entity` + `StackProtocol` is the recommended canonical API choice; owner must approve it.
- CLI only for init/status/diagnostics; no second application API.
- MCP is the sole alpha host boundary.
- Recommended retained types: raw evidence plus episode, with explicit raw-to-episode promotion. Owner must approve exact types.
- Get/search plus one export/restore path. Defer the overlapping checkpoint/dump systems.
- One `RequestContext`, one pure `PolicyEngine`, one provenance policy.
- One SQLite database per stack is the recommended isolation choice; owner must approve it instead of composite keys in one shared database.
- No model or provider is required for the alpha contract.

Why MCP first: it stays inside the Python package and avoids requiring a second host/runtime. If the owner rejects MCP, replace it explicitly; do not support multiple hosts in alpha.

## Defer Until Post-Alpha

- Hosted cloud, Supabase coupling, generic remotes, and sync expansion.
- Dashboard/web/mobile surfaces.
- OpenClaw and Claude Code host support.
- Postgres and third-party storage adapters.
- Cross-entity sharing, roles/groups, consent delegation, and trust networks.
- Memory types not explicitly retained in decision 0005.
- Model-driven processing and semantic-quality/provider claims.
- Checkpoint/dump duplication beyond the one chosen export/restore path.
- Pending/pondering, skills, subconscious/resonance, doctor patterns, and other advanced cognition.
- Billing, scale architecture, broad configurability, and compatibility shims.
- Performance promises until a workload and thresholds are approved.

## Bundle Order

1. Repository baseline and authority.
2. Product/API/policy decisions.
3. Test isolation, atomic writes, and migration atomicity.
4. Minimal workflow/privacy/provenance red contracts.
5. API consolidation and removal/quarantine of obsolete paths.
6. Pure owner-only policy/provenance core.
7. Stack isolation and canonical SDK boundary.
8. Parallel MCP and export/restore boundaries.
9. Installed-wheel E2E, docs, and release alignment.

## Required Tests

- Clean-checkout targeted/full test bootstrap.
- Installed-wheel CLI and MCP smoke.
- Three owner-approved minimal workflow scenarios.
- Owner-only privacy allow/deny/audit matrix across retained boundaries.
- Provenance mutation/repair/audit matrix for every retained memory type.
- Failed write, ID collision, partial migration, corrupt export, and omitted/forged identity paths.
- One real versioned MCP payload and installed-wheel canary.

## Human Steps Before Implementation

1. Confirm whether `dev-docs/` replaces `audits/` and whether it remains private.
2. Approve or replace the alpha slice.
3. Approve `Entity` + `Stack` and MCP, or replace each with one alternative.
4. Choose per-stack databases (recommended) or fully stack-scoped shared-database keys/queues.
5. Choose retained memory types plus create/update/delete/erase and raw-to-structured promotion rules.
6. Define the local identity trust root; recommended: the trusted local stack boundary supplies its owner principal and callers cannot choose another requester.
7. Supply or approve synthetic generation of the minimal fixtures in [source-docs/inputs-required.md](source-docs/inputs-required.md).
8. Declare no upgrade origins and use a clean breaking reset/export, or name each supported origin.
