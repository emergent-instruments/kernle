# Bundle: Request, Policy, and Provenance Core

Type: worker/backend

Owns: pure request identity, policy decision, access evaluation, provenance preservation/repair, and audit-event contracts. It does not thread every public surface.

Status: blocked on Bundles 01, 05, and 06.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema/API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Given the authenticated local owner and an existing/incoming retained record, one deterministic module decides allow, deny, preserve, erase, or repair and produces a reasoned audit event.

## Services

- None; policy must run offline and independently of SQLite.

## Libraries

- Python dataclasses/enums/protocols unless an existing local type already fits.

## Schema / API / Components

- `RequestContext`: authenticated requester, owner, trusted source capability, operation, and context needed by retained workflows.
- `PolicyDecision`: allow/deny/erase/repair, stable reason code, visible fields, audit event.
- Provenance: write-once source type/entity, append-only derivation/confidence history, explicit privileged repair.

## Implementation Notes

- Caller text such as `source="kernle:anything"` is metadata, never authentication.
- Omitted requester follows the approved contract; recommended behavior is fail closed.
- Defer roles/groups, delegated consent, public access, remote principals, and sync policy.
- Keep policy pure; storage-specific filtering compiles from its decision.

## Unit Tests

- Every privacy and provenance fixture case.
- Forged source/requester, omitted requester, attempted rewrite, approved erasure, and explicit repair.

## Integration Tests

- Storage-independent conformance cases only; SQLite persistence belongs to Bundle 08.

## E2E Tests

- Deferred to boundary bundles.

## Raw Fixtures

- `tests/fixtures/product/privacy-cases.yaml`
- `tests/fixtures/product/provenance-cases.yaml`

## Done

- Policy tests pass without a concrete storage backend.
- Stable reason codes cover every approved case.
- No public string can grant trusted internal status.
- Ordinary updates cannot erase protected provenance; repair is explicit and audited.
