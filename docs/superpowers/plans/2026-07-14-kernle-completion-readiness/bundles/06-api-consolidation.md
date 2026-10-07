# Bundle: API Consolidation

Type: worker/backend

Owns: removal/quarantine of duplicate `Kernle`/Entity/Stack paths, legacy mixins, exports, and contract tests before policy boundary work.

Status: blocked on Bundles 01 and 05. It must complete before policy is threaded through public boundaries.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Users learn and maintain one public read/write model, and an internal change cannot bypass policy through a legacy path.

## Services

- None.

## Libraries

- No new dependencies.

## API / Components

- Implement decision 0001 exactly.
- Recommended: `Entity` coordinates; `StackProtocol` owns memory contract; CLI/MCP call Entity; remove `Kernle` or reduce it to a thin non-owning facade.
- Remove obsolete examples, mixins, exports, and tests instead of compatibility shims.

## Implementation Notes

- Perform after the minimal behavior contracts exist and before policy threading, so agents secure only retained paths.
- Keep commits narrow by public contract slice.

## Unit Tests

- One canonical API export/constructor contract; removed paths fail import or are explicitly internal.

## Integration Tests

- Existing alpha workflows pass through the one retained path.

## E2E Tests

- A minimal canonical SDK smoke exercises the retained path; CLI/MCP installed-product proof belongs to Bundle 11.

## Raw Fixtures

- Minimal product workflows from Bundle 05.

## Done

- One public read path and one public write path remain.
- No duplicate retained public path or public raw-storage exposure remains; policy-bypass proof belongs to Bundles 08-10.
- Docs, typing, package exports, and tests name only the canonical API.
- Focused API contracts and all affected regression tests pass; Bundle 05's owned strict expected failures remain expected and unchanged.
