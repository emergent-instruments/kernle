# Spaces data model

`registry.db` in the Kernle home (`ARCH-009`) holds one table. Each space's own
data lives in its directory `spaces/<id>/`, which holds `kernle.db`, cards,
skills, and changelog. The other modules own those entities.

```mermaid
---
config:
  theme: dark
  flowchart:
    htmlLabels: true
    wrappingWidth: 760
  themeVariables:
    edgeLabelBackground: '#1f2c38'
  themeCSS: |
    .pip-entity { border-collapse: collapse; background: #111827; color: #e5edf5; text-align: left; }
    .pip-entity th, .pip-entity td { border: 1px solid #344556; padding: 8px 12px; text-align: left; }
    .pip-entity th { background: #263544; color: #e5edf5; }
    .pip-entity code { color: #e5edf5; font-family: monospace; }
    .badge { display: inline-block; white-space: nowrap; border: 1px solid; border-radius: 4px; padding: 2px 5px; margin: 2px; }
    .badge.unique { background: #163c2a; border-color: #6fd38a; color: #b8f5cb; }
---
flowchart TB
  SPACE["<table class='pip-entity'><tr><th colspan='4'>DATA-001 · space · registry.db</th></tr><tr><th>ATTRIBUTE</th><th>TYPE</th><th>KEY / RULE</th><th>INDEX BADGE</th></tr><tr><td><code>id</code></td><td><code>TEXT</code></td><td>PK · ULID · names the directory spaces/&lt;id&gt;/</td><td></td></tr><tr><td><code>kind</code></td><td><code>TEXT</code></td><td>Not null · 'user' or 'project' · one 'user' row per home</td><td></td></tr><tr><td><code>root_key</code></td><td><code>TEXT</code></td><td>Not null · 'user', 'git:&lt;normalized origin&gt;', or 'path:&lt;toplevel&gt;' (SEQ-003)</td><td><span class='badge unique'>[U1·1]</span></td></tr><tr><td><code>display_name</code></td><td><code>TEXT</code></td><td>Not null · shown in status and changelog</td><td></td></tr><tr><td><code>created_at</code></td><td><code>TEXT</code></td><td>Not null · ISO-8601 UTC</td><td></td></tr><tr><td><code>last_active_at</code></td><td><code>TEXT</code></td><td>Not null · ISO-8601 UTC · last capture or recall activity (SEQ-003) · drives idle deletion (retention SEQ-062)</td><td></td></tr><tr><th colspan='4'>INDEXES</th></tr><tr><td><span class='badge unique'>[U1]</span></td><td colspan='3'><code>space_root_key_key</code><br/>UNIQUE (root_key ASC)<br/>Supports SEQ-003: exactly one space per repository identity, and one personal space</td></tr></table>"]
  DIR["REFERENCE · space directory spaces/&lt;id&gt;/<br/>kernle.db holds DATA-002 to DATA-010, DATA-012, DATA-013<br/>cards/ and skills/ hold DATA-011 card files"]
  SPACE -->|"space.id names the directory · 1 space : 1 directory · deleting the space removes the directory (retention SEQ-062)"| DIR
  classDef entity fill:#111827,stroke:#8ea0b3,color:#e5edf5
  classDef ref fill:#111827,stroke:#8ea0b3,color:#b8c4d0,stroke-dasharray: 5 5
  class SPACE entity
  class DIR ref
```

Legend: green `U` = unique index. `·1` = first key column.

## Current rationale

- The registry is separate from space databases because resolving a request's
  space must happen before any space database is chosen. Idle-space deletion
  must also be able to delete a whole space without that space's database
  deleting its own registry entry.
- `root_key` is unique so concurrent first requests from two agents in a new
  repository resolve to the same space rather than splitting its memory.
- `last_active_at` lives in the registry because idle deletion considers every
  space, including spaces whose databases are never opened during an idle year.
