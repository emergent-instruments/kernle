# Bundle: Security Contract Foundation

Type: schema

Owns: three minimal alpha workflows, owner-only access cases, provenance cases, fixture validation, and failing security contract tests. It does not own semantic/model evaluation, host canaries, migrations, or runtime fixes.

Status: blocked on Bundle 01 and exactly three approved files: workflow acceptance, privacy cases, and provenance cases. Qualification/provider/host/migration material is not a dependency.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, Schema Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

The fleet has the minimum executable oracles required to implement owner-only privacy and raw-to-structured provenance safely.

## Services

- None.

## Libraries

- Existing pytest and JSON/YAML patterns. Prefer standard library or existing dependencies.

## Schema / API / Components

- Validate workflow, privacy, and provenance fixture structure before tests consume it.
- Limit the first matrix to the retained memory types and owner-only local trust boundary.
- Keep future roles/groups/remote consent out of these contracts.

## Implementation Notes

- Demonstrate red behavior first, then commit each known gap with a strict expected-failure marker and stable reason/bundle owner so the shared branch remains green.
- Do not weaken expected fixtures to match current code.

## Unit Tests

- Fixture validators reject missing identity, expected decision, provenance, or source labels.

## Integration Tests

- Contract matrix runs approved cases against the retained canonical SDK/Stack seam.

## E2E Tests

- Three canonical SDK workflows are expressed as red tests; installed-product E2E belongs to Bundle 11.

## Raw Fixtures

- `tests/fixtures/product/workflow-acceptance.yaml`
- `tests/fixtures/product/privacy-cases.yaml`
- `tests/fixtures/product/provenance-cases.yaml`

## Done

- Approved minimal workflow/security fixtures are tracked and validated.
- Known gaps are represented by strict expected-failure tests without runtime implementation changes; an unexpected pass fails the suite.
- Bundles 07-10 remove their owned expected-failure markers as behavior goes green.
- No semantic-quality, provider, migration, or host evidence blocks policy implementation.
