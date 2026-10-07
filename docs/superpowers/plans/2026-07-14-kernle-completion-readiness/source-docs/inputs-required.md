# Owner Inputs Required

Only the owner can settle product intent and authorize representative private material. Agents can generate synthetic cases after these choices are approved.

## Who Must Supply What

Owner-only input is limited to:

- approval/replacement of the alpha cut and decisions 0001-0006;
- confirmation of the `audits/` to `dev-docs/` intent/privacy;
- only if a later semantic/model claim is desired: permission to use fully synthetic qualification data or optional sanitized representative sessions;
- named supported upgrade origins, if any;
- optional selected-provider credential/budget only if a later model claim is desired;
- approval of generated acceptance/security fixtures before they become authority.

The fleet can draft all Markdown/YAML/JSONL from those answers, generate synthetic stacks and migration databases from public release artifacts, capture a local MCP fixture, and build validators/tests. The owner does not need to hand-author fixture files. Without representative owner review, synthetic cases can prove mechanics/security but cannot support a broad semantic-quality claim.

## Minimum Decisions

After Bundle 00 establishes the canonical docs location, add:

### `docs/product/alpha-brief.md`

Keep it to roughly one page:

- primary actor and beneficiary;
- job to be done;
- three canonical start-to-finish workflows;
- expected outcome and one measurable alpha success signal per workflow;
- unacceptable privacy, provenance, and data-loss failures;
- explicit in-scope and non-goals;
- selected host; recommended alpha answer is MCP only;
- local-only versus hosted deployment;
- supported Python/OS/provider/upgrade origins;
- release/version meaning.

### `docs/product/decisions/0001-public-api.md`

Choose one:

- recommended: `Entity` + `StackProtocol`, remove `Kernle` or keep only a very thin CLI facade;
- `Kernle` as the sole public facade delegating to Entity;
- a different single API, with a concrete reason.

Also decide whether CLI/MCP may write structured memories directly or may only capture raw evidence for later promotion.

### `docs/product/decisions/0002-sovereignty-and-promotion.md`

Define the manual raw-to-retained-type promotion, validation, failure/rollback, approval, and required provenance/audit evidence. Explicitly defer model-assisted/automatic promotion, pending behavior, and non-retained memory types.

### `docs/product/decisions/0003-access-policy.md`

For alpha, define owner-only access, export/restore, repair, and audit semantics. State whether an omitted requester is invalid or explicitly self. Recommended: omitted requester is invalid at every public boundary. Mark public/cross-entity access, roles/groups, delegated consent, remote, and sync policy deferred instead of designing them now.

### `docs/product/decisions/0004-stack-isolation.md`

Choose one database per stack (recommended for alpha) or a shared database with composite stack-scoped keys and stack-scoped auxiliary/search/sync tables. State collision, move/copy, import, and deletion semantics. Do not allow agents to preserve both designs.

### `docs/product/decisions/0005-memory-lifecycle.md`

Name the exact retained alpha memory types; recommended: raw and episode only. For each, define create, update, delete, hard erasure, export/restore, and provenance behavior. Explain how required lineage is removed when privacy erasure is legally/product-required. Mark every other type deferred rather than asking agents to maintain it in the alpha path.

### `docs/product/decisions/0006-identity-trust-root.md`

Define how the runtime obtains an authenticated owner/request source. Recommended local alpha: each stack has one owner principal supplied by the trusted local Stack/MCP process; public callers cannot set requester identity or trusted internal capability through arguments such as `source`. Name the trusted boundary, credential/capability representation, forgery behavior, and audit identity.

### `docs/product/surface-status.yaml`

One row per public Python method, CLI command, MCP tool, integration, and docs page:

```yaml
- surface: cli.note
  status: implemented # implemented | partial | planned | deferred | remove
  release: alpha
  authoritative_test: tests/e2e/test_cli_note.py
  notes: ""
```

## Product Test Material

### Workflow acceptance

Put `tests/fixtures/product/workflow-acceptance.yaml` in the tracked repository. Three cases are enough to unblock the minimal security contract: raw capture to retained structured promotion; owner get/search; owner export/restore plus denied/forged access. Each case needs persona, preconditions, steps, expected observable output, failure state, surface, and test path. Broader installed-product cases belong to Bundle 11.

### Privacy truth table

Put `tests/fixtures/product/privacy-cases.yaml` in the tracked repository. For retained alpha operations, include owner, requester, context, operation, expected allow/deny/audit decision, and reason. Include owner, omitted requester, forged requester/source, get, search, promotion, export, restore, and repair. Add roles/groups/consent/sync only if the owner rejects the owner-only cut.

### Provenance truth table

Put `tests/fixtures/product/provenance-cases.yaml` in the tracked repository. For each retained type, include existing lineage, attempted mutation, caller identity/capability, entry surface, fields that must survive, expected error/repair/erasure path, and audit event.

### Qualification sessions

Put sanitized tracked cases under:

```text
tests/fixtures/qualification/sessions/<case-id>/input.jsonl
tests/fixtures/qualification/sessions/<case-id>/expected.yaml
tests/fixtures/qualification/sessions/<case-id>/README.md
```

This is post-alpha, claim-conditional material and does not block security contract work. If semantic/model work is later approved, start with 10-20 multi-session cases and reach 30-50 before making a quality claim. Cover happy path; contradiction; stale belief; sensitive third-party data; denied access; prompt injection/memory poisoning; malformed model response; duplicate/replay; long-history budget truncation; offline/no-model. Expected files should assert invariants and acceptable bounds, not exact live-model prose.

Every case README must state origin, license/permission, redaction, whether synthetic, and the human reviewer/date.

### Search and retrieval labels

Put realistic sanitized records in `tests/fixtures/qualification/search/memories.jsonl` and 30-50 queries in `queries.jsonl`. Add `rubric.yaml` with the chosen retrieval metric, minimum threshold, privacy zero-tolerance, empty/abstain rule, aggregation, reviewer, and date; add a validator/test path. Each query needs graded relevant IDs and must-not-return privacy IDs. This material is required only before claiming semantic search quality, not for the minimal owner-only contract.

### Provider response shapes

For the one selected provider, add redacted recordings under `tests/fixtures/providers/<provider>/`: successful structured response, streaming chunks if supported, tool calls, malformed output, timeout, rate limit, and provider error. Include capture method/model/version and remove keys/request identifiers. These fixtures prove parsing/failure handling; live quality still uses the opt-in service.

### Migration snapshots

If upgrade compatibility is required, add sanitized SQLite databases under `tests/fixtures/migrations/<release>/` plus `manifest.yaml` with source package/schema version, target, row-count/content/provenance invariants, and generation instructions. Never use a real personal stack. If breaking reset is acceptable, say so in the alpha brief and delete compatibility requirements.

### Host payloads

For the one selected host, add real sanitized, versioned request/event/transcript fixtures:

- MCP: `tests/fixtures/mcp/<protocol-version>/`
- Claude Code: `tests/fixtures/claude-code/<version>/`
- OpenClaw: `integrations/openclaw/tests/fixtures/<version>/`

Record host version, capture command, redaction, and expected results.

## Private Material

Do not paste secrets or raw private conversations into chat or commit them.

Before placing private fixtures, add `tests/fixtures/private/` to `.gitignore`, verify with `git check-ignore`, then place local-only material there. Commit only a sanitized/example manifest under `tests/fixtures/private.example/`. Agents can transform private cases into synthetic equivalents locally if explicitly authorized.

## Credentials and Access

Keep values in an untracked permission-restricted file such as `~/.config/kernle/fleet.env`. Track variable names only in `.env.example` and `docs/product/platform-support.md`.

Add `docs/product/access-matrix.md` with one row per selected service/host: purpose, public/private URL or local path, account owner, required environment-variable names, verification command, allowed mutations, forbidden actions, and whether CI has equivalent access. Do not put values in this file.

For a one-provider live smoke, provide only the selected provider key and approve model, per-run budget, daily budget, timeout, and network allowance. Do not provide cloud/database credentials when cloud is deferred.

If `dev-docs/` remains a private submodule, ensure the local Git/SSH session and CI secret can read it. Record allowed mutations; read access does not authorize agents to publish or rewrite history.
