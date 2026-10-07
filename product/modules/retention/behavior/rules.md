# Retention rules

## RULE-040 Card retention decision

[SEQ-060](../sequences/sequences.md#seq-060-retention-sweep) applies this rule
to every card in the space on each sweep. Inputs come from `DATA-007 card`
(`kind`, `pinned`, `status`, `last_used_at`, `valid_to`), the sweep time, and
[settings.yaml](../contracts/settings.yaml).

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 380
  themeVariables:
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  IN["One card at sweep time 'now'"]
  P{"pinned = 1?"}
  S{"status = superseded?"}
  SX{"now - valid_to ≥ superseded_retention_days (90)?"}
  W["window = card_idle_days_preference_guardrail (365) when kind is preference or guardrail,<br/>else card_idle_days (180)<br/>idle_days = now - last_used_at<br/>strength = clamp(1 - idle_days / window, 0, 1)"]
  D{"idle_days ≥ window?"}
  DM{"strength &lt; dormant_below (0.5)?"}
  KEEP_PIN["KEEP · strength 1"]
  KEEP_SUP["KEEP · still superseded, never injected"]
  DELETE["DELETE · memory ApplyCardOp(delete), reason idle or superseded"]
  DORMANT["DORMANT · status dormant when currently active"]
  ACTIVE["ACTIVE · status active when currently dormant (it was used again)"]
  IN --> P
  P -->|Yes| KEEP_PIN
  P -->|No| S
  S -->|Yes| SX
  SX -->|Yes| DELETE
  SX -->|No| KEEP_SUP
  S -->|No| W --> D
  D -->|Yes| DELETE
  D -->|No| DM
  DM -->|Yes| DORMANT
  DM -->|No| ACTIVE
```

### Current rationale

- Idle time counts from the last use, not from creation. A memory that keeps
  helping should live indefinitely, and one that never helps should not
  survive just because it was once important.
- Preferences and guardrails get a longer window. A guardrail against a rare
  but costly mistake is idle by design between occurrences, and a preference
  can go untested for months while still being true.
- Dormancy at half the window takes a fading memory out of automatic recall
  before deleting it. Its cost to context ends early, while a deliberate
  search can still find it and a use brings it back.
- Pinned cards never expire. The owner's explicit choice outranks every
  heuristic.
