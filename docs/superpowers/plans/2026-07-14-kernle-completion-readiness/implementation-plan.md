# Implementation Plan

This is a compatibility pointer, not a duplicate plan.

- Product cut and dependency order: [README](README.md) and [alpha-simplification.md](alpha-simplification.md)
- Review evidence and test sufficiency: [readiness-audit.md](readiness-audit.md)
- Required owner decisions/data: [source-docs/inputs-required.md](source-docs/inputs-required.md)
- Work units: [bundles/](bundles/)
- Local and live services: [services/](services/)
- Fixture contract: [samples/README.md](samples/README.md)
- Cross-bundle contract ownership: [schemas/README.md](schemas/README.md)
- Implementation rules: [standards/codebase-guidelines.md](standards/codebase-guidelines.md)
- Independent review loop: [audits/bundle-audit-protocol.md](audits/bundle-audit-protocol.md)
- Operator state guidance: [ui-guidance.md](ui-guidance.md)
- Visual authority: [mockups/README.md](mockups/README.md)

The coordinator should maintain a dependency DAG, not a flat issue queue:

```mermaid
flowchart TD
  A["00 Repository baseline"] --> B["01 Product contract freeze"]
  B --> C["02 Atomic write disposition"]
  A --> D["03 Migration atomicity"]
  B --> E["04 Stack isolation"]
  D --> E
  B --> F["05 Security contract foundation"]
  F --> G["06 API consolidation"]
  G --> H["07 Policy and provenance core"]
  H --> I["08 SDK and Stack boundaries"]
  E --> I
  I --> J1["09 CLI and MCP boundaries"]
  I --> J2["10 Serialization boundaries"]
  C --> K["11 Release and qualification gate"]
  J1 --> K
  J2 --> K
  B -.-> O["99 OpenClaw repair (only if retained)"]
```
