# Samples and Qualification Fixtures

## Existing Material

- `tests/fixtures/golden_corpus/`: eight tracked files, 188 lines/about 5.8 KB.
- Inline deterministic pipeline/golden inputs and prompt-keyed mock responses.
- Many generated SQLite records and hand-built in-memory migration schemas.
- Mocked MCP, OpenClaw bridge, and cloud responses.

These are useful for deterministic regression and plumbing. They are not sufficient evidence of semantic quality, privacy/consent policy, real host compatibility, remote behavior, or supported-release migrations.

## Required Tracked Fixtures

| Path | Consumer | Required content |
|---|---|---|
| `tests/fixtures/product/workflow-acceptance.yaml` | installed-product E2E | canonical workflows and failure states |
| `tests/fixtures/product/privacy-cases.yaml` | policy and all boundary contracts | actors, subjects, consent, operations, decisions, reasons, audit |
| `tests/fixtures/product/provenance-cases.yaml` | provenance and import/sync contracts | existing/incoming lineage, expected preserve/reject/repair |
| `tests/fixtures/qualification/sessions/<id>/` | deterministic/live evals | multi-session JSONL, expected invariants, provenance/permission README |
| `tests/fixtures/qualification/search/` | search eval | records, queries, graded relevance, privacy exclusions |
| `tests/fixtures/providers/<provider>/` | provider parser/failure contracts | redacted success, stream/tool, malformed, timeout/rate-limit/error shapes |
| `tests/fixtures/migrations/<release>/` | migration E2E | sanitized DB plus manifest/invariants |
| `tests/fixtures/mcp/<version>/` | MCP contract | real sanitized protocol payloads |
| selected host fixture path | host integration | real sanitized event/config/transcript shapes |

## Structural Examples

- [privacy-cases.example.yaml](privacy-cases.example.yaml)
- [provenance-cases.example.yaml](provenance-cases.example.yaml)
- [qualification-input.example.jsonl](qualification-input.example.jsonl)
- [qualification-expected.example.yaml](qualification-expected.example.yaml)
- [migration-manifest.example.yaml](migration-manifest.example.yaml)

These examples illustrate file shape only. Their decisions and thresholds are not product authority.

## Negative Cases

Minimal alpha: omitted/forged requester/source; provenance rewrite/erasure; cross-stack ID collision; failed write; partial migration; corrupt export/restore; malformed MCP input; duplicate/replay. Post-alpha model/remote cases add prompt injection/memory poisoning, malformed model output, timeout/rate limit, long-history truncation, and provider/host failures only when those claims are reintroduced.

## Live Capture Instructions

1. Capture only from a synthetic test stack and supported host version.
2. Remove secrets, real names, tokens, filesystem identifiers, and private memory.
3. Preserve structure and relevant ordering; replace content with representative synthetic text.
4. Add a README naming version, capture method, redaction, permission/license, reviewer, and date.
5. Run a secret/PII check before commit.

## Private Cases

Before creating `tests/fixtures/private/`, add it to `.gitignore` and verify `git check-ignore`. Keep committed examples under `tests/fixtures/private.example/`; never copy private material into a report or issue.
