# Cross-Bundle Contracts and Schema Ownership

## Product Contracts

Bundle 01 owns alpha brief, surface status, platform support, API/write/access/isolation/trust-root/memory-lifecycle decisions, and workflow acceptance schema. No implementation bundle may reinterpret them.

## Policy Contracts

Bundle 07 owns `RequestContext`, `PolicyDecision`, reason codes, audit event shape, and provenance preservation/repair behavior. Boundary bundles consume these types; they do not fork local variants.

## SQLite Ownership

Bundle 03 owns migration atomicity/version advancement. Bundle 04 owns the one chosen stack-isolation model. Recommended: one DB per stack and no sync in alpha. No other bundle changes these paths or schema order.

## Boundary Ownership

- Bundle 06: canonical API removal/quarantine before policy threading.
- Bundle 08: Python SDK and Stack.
- Bundle 09: CLI and MCP.
- Bundle 10: one export/restore local-file path.
- Sync/remote is deferred and requires a separate future bundle.

## Fixture Schemas

Bundle 05 owns only minimal workflow, owner-only privacy, and provenance fixtures. Bundle 11 owns installed-product, MCP-host, upgrade, search, and optional model-quality evidence. Expected decisions are owner-approved input and cannot be changed merely to make implementation pass.

## Migration Order

1. Make the migration runner fail-fast and version advancement atomic.
2. Decide stack isolation; recommended one DB per stack plus a breaking reset/export.
3. Add collision/failure fixtures and implement the chosen path.
4. Only add historical upgrade migrations if Bundle 01 names a supported origin.
5. Add policy/audit persistence only after Bundle 07 freezes its contract.
