# Bundle: Serialization Policy Boundaries

Type: worker/backend

Owns: one local export/restore path with policy mediation, validation, and redaction. Other dump/checkpoint/import variants plus sync/remote are deferred.

Status: blocked on Bundle 08.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

An export or import cannot reveal, widen, erase, or forge memory access and provenance outside the authenticated request's authority.

## Services

- Local files only for alpha; remote sync is out of scope.

## Libraries

- Existing serializers/importers and standard JSON/Markdown tools.

## API / Components

- Require request context for the one retained export/restore preview/commit path.
- Preserve or deliberately redact access and provenance metadata.
- Separate maintenance repair from normal import.

## Implementation Notes

- Never read storage tables directly from a public serializer without policy mediation.
- Restore must validate fully before commit and be atomic.

## Unit Tests

- Redaction, denied export, forged restore provenance, partial failure, and repair authorization.

## Integration Tests

- Round trip approved records; prove denied records never appear and lineage survives.

## E2E Tests

- Operator exports, inspects, restores, and is denied an unauthorized export using temporary files.

## Raw Fixtures

- Workflow/privacy/provenance fixtures plus generated malformed exports.

## Done

- Every public serialization/import path is policy-mediated.
- No unauthorized content or metadata leaks.
- Normal restore cannot rewrite protected provenance.
- Remote/sync claims are explicitly deferred or receive a separate reviewed bundle.
