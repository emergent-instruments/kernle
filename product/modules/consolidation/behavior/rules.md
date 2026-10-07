# Consolidation rules

[SEQ-031](../sequences/sequences.md#seq-031-sleep-run) applies these rules
during every sleep run, in this order: RULE-020 chooses which episodes to
replay, RULE-021 chooses what each replayed episode may become, and RULE-022
decides whether each proposed change applies, waits for the owner, or is
refused. Parameters are in [settings.yaml](../contracts/settings.yaml).

## RULE-020 Replay selection

Inputs: pending episodes from `DATA-004 episode [P1]`, each episode's
`salience`, `label`, and `summary_vec`, the space's episodes from the last 30
days, and whether the episode contains an `explicit_remember` event
(`DATA-002`).

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 360
  themeVariables:
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  IN["Each pending episode: salience, label, summary_vec, explicit_remember present"]
  R{"Contains an explicit_remember event?"}
  B{"label = boredom or salience &lt; replay_min_salience (0.35)?"}
  NEED["need = 0.5 + 0.5 × min(1, related_sessions_30d / need_sessions_saturation)<br/>related_sessions_30d = distinct sessions in the last 30 days with an episode<br/>whose cosine to this summary_vec ≥ need_similarity<br/>priority = salience × need"]
  ORDER["Replay order: explicit_remember episodes first,<br/>then priority descending, up to replay_cap_per_run episodes"]
  REPLAY["REPLAY · extraction in SEQ-031"]
  REINF["REINFORCE_ONLY · match active cards, no extraction"]
  WAIT["WAIT · stays pending for the next run (over the cap)"]
  IN --> R
  R -->|Yes| ORDER
  R -->|No| B
  B -->|Yes| REINF
  B -->|No| NEED --> ORDER
  ORDER -->|within the cap| REPLAY
  ORDER -->|beyond the cap| WAIT
```

### Current rationale

- Priority multiplies salience by need, as in prioritized replay: an
  experience is worth consolidating when it mattered and when similar
  situations keep coming up.
- Explicit remember requests always replay. The person or agent asked, and
  skipping a request because it scored low would break trust in remember.
- Low-salience and boredom episodes only reinforce existing memories. Routine
  success should strengthen what is already known, not create new memories
  that dilute the important ones.

## RULE-021 Filing route

The episode's label from
[appraisal RULE-011](../../appraisal/behavior/rules.md#rule-011-functional-emotion-label)
decides which card kinds and operations the Dreamer may propose. A proposal
outside its route is refused at validation in SEQ-031.

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 420
  themeVariables:
    edgeLabelBackground: '#1f2c38'
---
flowchart LR
  L["Episode label (RULE-011)<br/>explicit_remember present?"]
  L -->|explicit_remember present| K0["Kind requested by the remember call,<br/>else one of fact, preference, lesson, guardrail, open-loop<br/>op: add, or update of a matching card"]
  L -->|alarm| K1["guardrail · 'before X, check Y' with triggers (tools, command patterns, paths)<br/>op: add, or update of a matching guardrail"]
  L -->|frustration| K2["lesson · pitfall, fix, and why<br/>op: add, update, or merge"]
  L -->|correction| K3["preference · scope user for the person's preferences,<br/>project for project conventions<br/>op: add, update, or supersede"]
  L -->|relief| K4["skill when a repeatable multi-step procedure succeeded, else lesson<br/>op: add or update"]
  L -->|surprise| K5["fact · op: add, or supersede the contradicted fact"]
  L -->|curiosity| K6["open-loop · op: add, or update the matching open loop"]
  L -->|approval| K7["no new card · op: reinforce or update confidence<br/>of the preference or skill that was followed"]
  L -->|distrust| K8["lesson about the source's reliability<br/>op: add"]
  L -->|boredom| K9["not replayed (RULE-020 REINFORCE_ONLY)"]
  RET["Every replayed route may also propose retire<br/>of an active open-loop that this episode resolved"]
  K1 & K2 & K3 & K4 & K5 & K7 & K8 -.-> RET
```

### Current rationale

- The label decides the kind of memory. That is how emotion files memory: a
  near-disaster should become a pre-action guardrail, a correction a
  preference, and a hard-won success a procedure. The same experience filed
  as a generic note would not change behavior at the moment it matters.
- Approval reinforces instead of creating. Praise confirms something already
  known, and a new card for it would duplicate rather than strengthen.

## RULE-022 Change policy

Applied to every validated card operation before it is applied. Inputs:
operation type and card kind from the proposal, `evidence_trust` (the highest
`DATA-002 event.trust` among the operation's cited events, ordered principal >
first_party > agent_inferred > third_party), and, for updates, the target card's
current author and `pinned` value (`DATA-007`, `DATA-008`).

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 340
  themeVariables:
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  IN["Validated card operation<br/>op, kind, evidence_trust, target author, target pinned"]
  P1{"op is update, merge, or supersede, and the target's current version<br/>is owner-authored or the target is pinned?"}
  P2{"kind is guardrail, lesson, preference, or skill,<br/>and evidence_trust = third_party?"}
  P3{"kind = preference and evidence_trust ≠ principal?"}
  REVIEW_OWNER["REVIEW · owner content is never changed without the owner"]
  QUAR["QUARANTINE · held for review and marked as<br/>supported only by outside content"]
  REVIEW_PREF["REVIEW · a preference must come from the person"]
  AUTO["AUTO · apply now through memory ApplyCardOp"]
  IN --> P1
  P1 -->|Yes| REVIEW_OWNER
  P1 -->|No| P2
  P2 -->|Yes| QUAR
  P2 -->|No| P3
  P3 -->|Yes| REVIEW_PREF
  P3 -->|No| AUTO
```

REVIEW and QUARANTINE operations wait in the review queue
([SEQ-032](../sequences/sequences.md#seq-032-resolve-a-held-change)) and expire
unapplied after `review_expiry_days`.

### Current rationale

- Behavior-changing memories need first-party or owner support. Persistent
  prompt injection works by getting an agent to "remember" an instruction it
  read somewhere. Quarantining what only outside content supports blocks that
  path without blocking memories of things that actually happened.
- Preferences require the person's own words. An agent inferring what its
  owner likes, then acting on it every session, is how memory drifts away
  from the person it serves.
- Owner-authored or pinned cards are never rewritten automatically. When the
  owner has spoken, sleep proposes and the owner decides.
- Everything else applies automatically. Owner attention is scarce, the
  changelog shows every change, and any card can be edited or forgotten later.
