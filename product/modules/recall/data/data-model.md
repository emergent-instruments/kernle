# Recall data model

Both entities live in each space's `kernle.db` (`ARCH-009`). `DATA-013` is
derived from cards and can be rebuilt at any time. `DATA-012` is the usage
record that reconsolidation and gut-check history depend on. Retention deletes
`recall_log` rows after `recall_log_retention_days`, and with their card
([retention SEQ-060, SEQ-061](../../retention/sequences/sequences.md#public-boundary)).

```mermaid
---
config:
  theme: dark
  flowchart:
    htmlLabels: true
    wrappingWidth: 860
  themeVariables:
    edgeLabelBackground: '#1f2c38'
  themeCSS: |
    .pip-entity { border-collapse: collapse; background: #111827; color: #e5edf5; text-align: left; }
    .pip-entity th, .pip-entity td { border: 1px solid #344556; padding: 8px 12px; text-align: left; }
    .pip-entity th { background: #263544; color: #e5edf5; }
    .pip-entity code { color: #e5edf5; font-family: monospace; }
    .badge { display: inline-block; white-space: nowrap; border: 1px solid; border-radius: 4px; padding: 2px 5px; margin: 2px; }
    .badge.index { background: #162c48; border-color: #6ea8ff; color: #c6ddff; }
---
flowchart TB
  LOG["<table class='pip-entity'><tr><th colspan='4'>DATA-012 · recall_log · space kernle.db</th></tr><tr><th>ATTRIBUTE</th><th>TYPE</th><th>KEY / RULE</th><th>INDEX BADGE</th></tr><tr><td><code>id</code></td><td><code>TEXT</code></td><td>PK · ULID</td><td></td></tr><tr><td><code>session_id</code></td><td><code>TEXT</code></td><td>Not null · runtime session</td><td><span class='badge index'>[I1·1]</span></td></tr><tr><td><code>tier</code></td><td><code>TEXT</code></td><td>Not null · index, cue, gut_advise, speed_bump, search, read, or feedback</td><td></td></tr><tr><td><code>card_id</code></td><td><code>TEXT</code></td><td>Not null · FK to card.id</td><td><span class='badge index'>[I2·1]</span></td></tr><tr><td><code>action_key</code></td><td><code>TEXT</code></td><td>Nullable · hash of tool_name and normalized tool_input · gut-check tiers only · RULE-032</td><td></td></tr><tr><td><code>outcome</code></td><td><code>TEXT</code></td><td>Nullable · helped, hurt, irrelevant, wrong, or outdated (SEQ-054)</td><td></td></tr><tr><td><code>created_at</code></td><td><code>TEXT</code></td><td>Not null · ISO-8601 UTC · consolidation maps rows to episodes by session and time</td><td><span class='badge index'>[I1·2]</span></td></tr><tr><td><code>expires_at</code></td><td><code>TEXT</code></td><td>Not null · created_at + recall_log_retention_days</td><td></td></tr><tr><th colspan='4'>INDEXES</th></tr><tr><td><span class='badge index'>[I1]</span></td><td colspan='3'><code>recall_log_session_idx</code><br/>(session_id ASC, created_at ASC)<br/>Supports SEQ-051 shown-this-session exclusion, SEQ-052 gut-check history inside the hook deadline, and consolidation SEQ-031 episode mapping</td></tr><tr><td><span class='badge index'>[I2]</span></td><td colspan='3'><code>recall_log_card_idx</code><br/>(card_id ASC)<br/>Supports retention SEQ-061: delete a card's usage rows with the card</td></tr></table>"]
  IDX["<table class='pip-entity'><tr><th colspan='4'>DATA-013 · card search index · card_fts, card_vector, index_state</th></tr><tr><th>ATTRIBUTE</th><th>TYPE</th><th>KEY / RULE</th><th>INDEX BADGE</th></tr><tr><td><code>card_fts.card_id</code></td><td><code>TEXT</code></td><td>FTS5 virtual table, unindexed column · one row per active or dormant card</td><td></td></tr><tr><td><code>card_fts.name, description, body</code></td><td><code>TEXT</code></td><td>FTS5 indexed text · RULE-030 lexical</td><td></td></tr><tr><td><code>card_vector.card_id</code></td><td><code>TEXT</code></td><td>PK · FK to card.id</td><td></td></tr><tr><td><code>card_vector.model_id</code></td><td><code>TEXT</code></td><td>Not null · a vector is current only when it equals embedding.model_id</td><td></td></tr><tr><td><code>card_vector.version</code></td><td><code>INTEGER</code></td><td>Not null · card version the vector was computed from</td><td></td></tr><tr><td><code>card_vector.vector</code></td><td><code>BLOB</code></td><td>Not null · float32 array</td><td></td></tr><tr><td><code>index_state.ledger_cursor</code></td><td><code>INTEGER</code></td><td>Not null · last processed card_ledger.id (SEQ-055)</td><td></td></tr><tr><td><code>index_state.model_id</code></td><td><code>TEXT</code></td><td>Not null · a difference triggers a full rebuild</td><td></td></tr></table>"]
  CARD["REFERENCE · DATA-007 · card<br/>owned by memory"]
  CARD -->|"recall_log.card_id = card.id · 1 card : 0..many rows · deleted with the card"| LOG
  CARD -->|"card_fts.card_id and card_vector.card_id = card.id · 1 card : 0..1 row each · removed for superseded and deleted cards"| IDX
  classDef entity fill:#111827,stroke:#8ea0b3,color:#e5edf5
  classDef ref fill:#111827,stroke:#8ea0b3,color:#b8c4d0,stroke-dasharray: 5 5
  class LOG,IDX entity
  class CARD ref
```

Legend: blue `I` = ordinary index. Numeric suffixes are ordered key positions.

## Current rationale

- `recall_log` keeps the card and the session but never the prompt or tool
  input text. Gut-check history needs only a hash of the action. Storing the
  prompts recall responded to would duplicate the most sensitive content
  outside capture's retention.
- The search index stores the model it was built with. Changing the embedding
  model must never mix incompatible vectors, and a rebuild from files is cheap
  for a personal store.
