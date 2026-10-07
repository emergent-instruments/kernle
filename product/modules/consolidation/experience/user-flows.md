# Consolidation user flows

## FLOW-003 Review memory changes

`kernle log` reads the changelog projection that memory writes after each run
([memory SEQ-042](../../memory/sequences/sequences.md#seq-042-write-the-changelog)).
`kernle review` resolves held changes
([SEQ-032](../sequences/sequences.md#seq-032-resolve-a-held-change)).
`kernle sleep` starts a manual run
([SEQ-030](../sequences/sequences.md#seq-030-schedule-maintenance-runs)).

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 320
  themeVariables:
    clusterBkg: '#18222d'
    clusterBorder: '#344556'
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  START(["ACTOR-001 wants to know what Kernle learned"])

  subgraph S_LOG["SURFACE · kernle log"]
    L1["Show recent runs, newest first: space, time, learned N,<br/>updated N, forgot N, held for review N, budget note"]
    L2["Show a run's entries: card name and kind, change, why it mattered<br/>(label and top appraisal reasons), and source sessions"]
    L_EMPTY["Show no runs yet · when the next sleep is due"]
  end

  subgraph S_SLEEP["SURFACE · kernle sleep"]
    SL1["Show run progress: appraising, judging, replaying N of M"]
    SL2["Show run summary in the same shape as a kernle log run"]
    SL_BUSY["Show a run is already in progress · waits, then shows its summary"]
    SL_BUDGET["Show daily Dreamer budget reached · remaining episodes wait until tomorrow"]
  end

  subgraph S_REVIEW["SURFACE · kernle review"]
    R1["Show each held change: proposed card, kind, space,<br/>why it is held (owner content, preference not stated by you,<br/>or supported only by outside content), and cited evidence"]
    R2{"Accept · Edit · Reject · Skip"}
    R_DONE["Show nothing left to review"]
    R_CONFLICT["Show the card changed since the proposal,<br/>with the current card beside the proposal"]
    R_RO["Show held changes read-only:<br/>decisions need an interactive terminal"]
  end

  START -->|kernle log| L1
  L1 -->|choose a run| L2
  L1 -->|no runs| L_EMPTY
  START -->|kernle sleep| SL1
  SL1 --> SL2
  SL1 -->|run already in progress| SL_BUSY
  SL1 -->|budget reached| SL_BUDGET
  START -->|kernle review| R1
  START -->|kernle review without a terminal, e.g. from an agent tool| R_RO
  R1 --> R2
  R2 -->|Accept, applied| R1
  R2 -->|Accept, card changed| R_CONFLICT
  R_CONFLICT --> R2
  R2 -->|Edit: $EDITOR opens, saved text applied as owner content| R1
  R2 -->|Reject or Skip| R1
  R1 -->|queue empty| R_DONE
```

## Current rationale

- The log leads with counts, then reasons. Owners skim for "did anything
  surprising happen", and the reason ("held because supported only by outside
  content") is what lets them judge a change quickly.
- Review shows why each change is held. Different holds call for different
  owner judgments: protecting their own text, confirming a preference, or
  spotting a manipulation attempt.
- A conflict shows the current card beside the proposal rather than failing.
  The owner can then decide with full context.
