# Consolidation data model

Both tables live in each space's `kernle.db` (`ARCH-009`). Consolidation owns
them. Retention deletes decided `card_op` rows after `card_op_retention_days`
([retention SEQ-060](../../retention/sequences/sequences.md#seq-060-retention-sweep)).

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
    .badge.partial { background: #302047; border-color: #c58aff; color: #e4caff; }
---
flowchart TB
  RUN["<table class='pip-entity'><tr><th colspan='4'>DATA-005 · maintenance_run · space kernle.db</th></tr><tr><th>ATTRIBUTE</th><th>TYPE</th><th>KEY / RULE</th><th>INDEX BADGE</th></tr><tr><td><code>id</code></td><td><code>TEXT</code></td><td>PK · ULID</td><td></td></tr><tr><td><code>kind</code></td><td><code>TEXT</code></td><td>Not null · 'sleep' or 'retention'</td><td></td></tr><tr><td><code>trigger</code></td><td><code>TEXT</code></td><td>Not null · activity, import, manual, nightly, retention_daily, or volume</td><td></td></tr><tr><td><code>state</code></td><td><code>TEXT</code></td><td>Not null · 'running', 'completed', 'failed', or 'abandoned'</td><td><span class='badge partial'>[P1·where]</span></td></tr><tr><td><code>started_at</code></td><td><code>TEXT</code></td><td>Not null · ISO-8601 UTC · selects today's runs for the token allowance</td><td></td></tr><tr><td><code>finished_at</code></td><td><code>TEXT</code></td><td>Nullable</td><td></td></tr><tr><td><code>dreamer_tokens</code></td><td><code>INTEGER</code></td><td>Not null · default 0 · provider-reported input plus output tokens · counts against daily_token_budget</td><td></td></tr><tr><td><code>budget_exhausted</code></td><td><code>INTEGER</code></td><td>Not null · 0 or 1 · shown in kernle status</td><td></td></tr><tr><td><code>counters</code></td><td><code>TEXT</code></td><td>Not null · JSON: episodes appraised, judged, replayed, reinforced, and ops by state · shown in status and changelog</td><td></td></tr><tr><td><code>error</code></td><td><code>TEXT</code></td><td>Nullable · failure reason</td><td></td></tr><tr><th colspan='4'>INDEXES</th></tr><tr><td><span class='badge partial'>[P1]</span></td><td colspan='3'><code>maintenance_run_one_running_idx</code><br/>UNIQUE ((1)) WHERE state = 'running'<br/>Supports SEQ-030: at most one running maintenance run per space</td></tr></table>"]
  OP["<table class='pip-entity'><tr><th colspan='4'>DATA-006 · card_op · space kernle.db</th></tr><tr><th>ATTRIBUTE</th><th>TYPE</th><th>KEY / RULE</th><th>INDEX BADGE</th></tr><tr><td><code>id</code></td><td><code>TEXT</code></td><td>PK · ULID · idempotency key for memory ApplyCardOp</td><td></td></tr><tr><td><code>run_id</code></td><td><code>TEXT</code></td><td>Not null · FK to maintenance_run.id</td><td></td></tr><tr><td><code>episode_id</code></td><td><code>TEXT</code></td><td>Nullable · the replayed episode</td><td></td></tr><tr><td><code>op</code></td><td><code>TEXT</code></td><td>Not null · add, update, merge, supersede, retire, reinforce, noop, or delete (held agent forget)</td><td></td></tr><tr><td><code>kind</code></td><td><code>TEXT</code></td><td>Not null · fact, preference, lesson, guardrail, skill, or open-loop</td><td></td></tr><tr><td><code>target_card_id</code></td><td><code>TEXT</code></td><td>Nullable · card changed by update, merge, supersede, or reinforce</td><td></td></tr><tr><td><code>target_version</code></td><td><code>INTEGER</code></td><td>Nullable · expected card.current_version (COORD-002)</td><td></td></tr><tr><td><code>proposal</code></td><td><code>TEXT</code></td><td>Not null · JSON: name, description, body, triggers, scope, confidence</td><td></td></tr><tr><td><code>cited_ids</code></td><td><code>TEXT</code></td><td>Not null · JSON array of event and episode ids</td><td></td></tr><tr><td><code>evidence_trust</code></td><td><code>TEXT</code></td><td>Not null · highest cited event trust · RULE-022 input</td><td></td></tr><tr><td><code>policy</code></td><td><code>TEXT</code></td><td>Not null · auto, review, quarantine, or reject (RULE-022)</td><td></td></tr><tr><td><code>state</code></td><td><code>TEXT</code></td><td>Not null · applying, applied, conflict, refused, pending_review, accepted, rejected, or expired</td><td><span class='badge partial'>[P1·where]</span></td></tr><tr><td><code>reason</code></td><td><code>TEXT</code></td><td>Nullable · refusal or quarantine explanation shown in review</td><td></td></tr><tr><td><code>created_at</code></td><td><code>TEXT</code></td><td>Not null · ISO-8601 UTC · starts the review expiry window</td><td><span class='badge partial'>[P1·1]</span></td></tr><tr><td><code>decided_at</code></td><td><code>TEXT</code></td><td>Nullable · starts card_op_retention_days</td><td></td></tr><tr><th colspan='4'>INDEXES</th></tr><tr><td><span class='badge partial'>[P1]</span></td><td colspan='3'><code>card_op_review_queue_idx</code><br/>(created_at ASC) WHERE state = 'pending_review'<br/>Supports SEQ-032: list held changes oldest first and expire them</td></tr></table>"]
  EP["REFERENCE · DATA-004 · episode<br/>owned by appraisal"]
  CARD["REFERENCE · DATA-007 · card<br/>owned by memory"]
  RUN -->|"card_op.run_id = maintenance_run.id · 1 run : 0..many ops"| OP
  EP -->|"card_op.episode_id = episode.id · 1 episode : 0..many ops · nullable for review edits"| OP
  CARD -->|"card_op.target_card_id = card.id · 1 card : 0..many ops · null for add"| OP
  classDef entity fill:#111827,stroke:#8ea0b3,color:#e5edf5
  classDef ref fill:#111827,stroke:#8ea0b3,color:#b8c4d0,stroke-dasharray: 5 5
  class RUN,OP entity
  class EP,CARD ref
```

Legend: purple `P` = partial or expression index. `·1` = first key column.
`·where` = column used only in the predicate.

## Current rationale

- Daily token usage is the sum of run rows. The budget then needs no separate
  counter that could drift from the runs that actually spent the tokens.
- `card_op` keeps every proposal, including refusals and expirations, until
  its retention window ends. The changelog and review can explain what sleep
  decided, and a version conflict can be retried from the recorded proposal.
- The one-running index has a constant key with a predicate. The invariant is
  "one running row per space database", not per any column value.
