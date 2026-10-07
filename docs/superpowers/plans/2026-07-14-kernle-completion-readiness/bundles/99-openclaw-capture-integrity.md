# Bundle: OpenClaw Capture Integrity or Removal

Type: service

Owns: `integrations/openclaw/`, its CI/package contract, and public support labelling. It does not change Python memory semantics.

Status: optional and blocked on Bundle 01's explicit decision to retain OpenClaw. It is non-gating when MCP is selected.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

If retained, an intercepted memory write is blocked only after durable capture into the correct session stack. If deferred, users cannot mistake OpenClaw for a supported alpha surface.

## Services

- A pinned supported OpenClaw host/API source for real compatibility testing.

## Libraries

- Use official/pinned OpenClaw types or a versioned captured contract if available.
- Add a minimal working lint dependency/config only if OpenClaw remains supported.

## API / Components

- Resolve stack identity at the event boundary, not once with empty context.
- Treat `bridge.raw(...) === false` as capture failure and permit/fail-safe the original write with an explicit error.
- Validate clean package contents include the built entry point; `dist/` is ignored while npm `files` requires it.

## Implementation Notes

- TDD the current loss path before code.
- Do not silently fall back to a different stack.
- Removal is simpler than maintaining a second runtime without host evidence.

## Unit Tests

- Failed/timeout/missing CLI capture does not block the native write.
- Successful capture blocks exactly once.
- Two session contexts resolve to two correct stack IDs.

## Integration Tests

- Clean `npm ci`, test, lint, build, and `npm pack --dry-run` include the entry point.

## E2E Tests

- Real supported-host fixture/canary intercepts and retrieves one synthetic memory.

## Raw Fixtures

- `integrations/openclaw/tests/fixtures/<version>/`; blocked until supplied if support is retained.

## Done

- No failed capture can destroy the original write.
- No event can be routed to a default stack when session identity exists.
- Node CI validates test, lint, build, and package contents.
- Support docs name the proven host version, or all support claims are removed/deferred.
