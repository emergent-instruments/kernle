# Consolidation sequences

Consolidation schedules maintenance runs and orchestrates sleep. It owns
[DATA-005 and DATA-006](../data/data-model.md), the
[rules](../behavior/rules.md), and the review queue. Rust module `sleep` owns
these processes. Parameters are in [settings.yaml](../contracts/settings.yaml).

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [RequestMaintenance](#seq-030-schedule-maintenance-runs) | capture SEQ-011 and SEQ-012, `kernle sleep`, the scheduler | Schedules or starts a run. At most one maintenance run per space at a time. Requests during a run schedule a follow-up |
| [Sleep run](#seq-031-sleep-run) | SEQ-030 | Appraises, judges, replays, applies or holds card changes, sweeps retention, refreshes the index, and writes the changelog. Partial progress is kept. A failure leaves unprocessed work pending |
| [Resolve a held change](#seq-032-resolve-a-held-change) | `kernle review`, the daily expiry pass | Accept applies the change through memory ApplyCardOp. Reject and expiry never apply it |

## SEQ-030 Schedule maintenance runs

Rust function `sleep::scheduler` owns scheduling. It runs inside `kernled`
(`ARCH-008`), started by spaces SEQ-004.

**DCL:** 3 (product default)

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
    noteFontSize: 14
    noteFontFamily: Arial
  themeCSS: |
    .noteText { font-family: Arial !important; font-size: 14px !important; }
  themeVariables:
    noteBkgColor: '#242626'
    noteTextColor: '#e5edf5'
---
sequenceDiagram
  autonumber
  participant T as Trigger source
  participant SC as ARCH-008 scheduler
  participant DB as ARCH-009 space kernle.db

  Note over SC: At daemon start, every space's runs left in state 'running' are marked 'abandoned'<br/>UPDATE · DATA-005 maintenance_run state 'abandoned' WHERE state = 'running'<br/>When the last retention run started more than 24 hours ago, retention_daily is requested at once
  T->>SC: RequestMaintenance(space, trigger)
  Note right of SC: INPUT · trigger: activity (capture SEQ-011), import (capture SEQ-012),<br/>manual (kernle sleep), nightly (sleep_time local),<br/>retention_daily (retention_time local, every space), volume (event count)
  alt trigger = activity
    SC->>SC: due_at(space, sleep) = now + session_debounce_minutes
    Note over SC: Each new activity pushes due_at later (debounce)
  else trigger = volume, import, or manual
    SC->>SC: due_at(space, sleep) = now
  else trigger = nightly
    SC->>DB: Check for pending work
    Note right of DB: READ · DATA-002 event [P1] (any unsegmented)<br/>DATA-004 episode [P1] (any pending replay)
    SC->>SC: due_at(space, sleep) = now only when work is pending
  else trigger = retention_daily
    SC->>SC: due_at(space, retention) = now
  end
  loop When a due_at passes
    alt The space's in-process run guard is held
      Note over SC: Keep due_at so a follow-up runs after the current run completes
    else Guard free
      SC->>DB: Start the run
      Note right of DB: INSERT · DATA-005 maintenance_run<br/>WRITE · kind sleep or retention, trigger, state 'running', started_at<br/>CONSTRAINT · [P1] maintenance_run_one_running_idx
      alt Insert conflicts on [P1]
        Note over SC,DB: Another run is still marked running · retry at the next tick
      else Started
        SC->>SC: Run SEQ-031 (sleep) or retention RetentionSweep (retention SEQ-060)
        SC->>DB: Finish the run
        Note right of DB: UPDATE · DATA-005 maintenance_run<br/>WRITE · state 'completed' or 'failed', finished_at, counters, error
        opt Manual trigger
          SC-->>T: Stream the run summary to kernle sleep
        end
      end
    end
  end
```

### Current rationale

- Activity is debounced. Sleeping after every turn would spend the Dreamer
  budget on half-finished work, and episodes cannot close until the session
  moves on anyway.
- The nightly trigger runs only when work is pending, so idle spaces cost
  nothing. Retention runs daily for every space, including idle ones, because
  deletion must happen on schedule whether or not anyone uses the space.
- One run per space is enforced by the daemon's in-process guard, backed by
  `[P1]`. Only one daemon can run per home (spaces SEQ-004 file lock), and the
  index catches any second writer. A persisted lease would add expiry and
  fencing logic that a single process does not need.

## SEQ-031 Sleep run

Started by SEQ-030 for a space. Rust function `sleep::run()` owns the order.

**DCL:** 3 (product default)

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
    noteFontSize: 14
    noteFontFamily: Arial
  themeCSS: |
    .noteText { font-family: Arial !important; font-size: 14px !important; }
  themeVariables:
    noteBkgColor: '#242626'
    noteTextColor: '#e5edf5'
---
sequenceDiagram
  autonumber
  participant S as ARCH-008 sleep run
  participant DB as ARCH-009 space kernle.db
  participant N as ARCH-010 native memory files
  participant LM as ARCH-011 Dreamer provider

  S->>S: allowance = daily_token_budget - tokens used today
  Note over S: INPUT · tokens used today = sum of DATA-005 maintenance_run.dreamer_tokens<br/>for runs started today in every space<br/>dreamer.provider = 'none' gives allowance 0
  S->>S: appraisal AppraisePending(space) (appraisal SEQ-020)
  opt allowance > 0
    S->>S: appraisal JudgeEpisodes(space, judge_share × allowance) (appraisal SEQ-021)
  end
  S->>DB: Load pending episodes and select with RULE-020
  Note right of DB: READ · DATA-004 episode<br/>ACCESS · DATA-004 episode [P1] episode_replay_queue_idx
  loop Each REINFORCE_ONLY episode
    S->>S: recall NearestCards(summary_vec) (recall SEQ-055)
    alt A card's cosine ≥ reinforce_similarity
      S->>S: memory ApplyCardOp(reinforce, card) (memory SEQ-040)
      S->>DB: episode.replay_state = 'reinforced'
    else No close card
      S->>DB: episode.replay_state = 'skipped'
    end
  end
  loop Each REPLAY episode in RULE-020 order
    alt Episode has explicit_remember and (allowance = 0 or the runtime's dreamer_allowed is false)
      S->>S: Propose add with the remembered text as body, kind as requested or 'fact'
      Note over S: Cited events: the explicit_remember event plus user messages in the episode
    else allowance > 0 and dreamer_allowed for the episode's runtime
      S->>DB: Load the evidence excerpt and the cards recalled during the episode
      Note right of DB: READ · DATA-002 event [I1] (evidence_first_seq to last_seq)<br/>READ · DATA-012 recall_log [I1] rows for this session created between the episode's first and last event ts
      S->>S: recall NearestCards for related active cards (related_cards_k)
      S->>N: Read native memory for the project (read-only)
      Note right of N: Claude Code project memory files and CLAUDE.md or AGENTS.md<br/>OpenClaw MEMORY.md for the personal space<br/>Used only to avoid duplicating what native memory holds
      S->>LM: Extraction request (excerpt, appraisal, RULE-021 route, related cards, native lines, recalled cards)
      Note right of LM: Returns JSON: operations (op, kind, target card and version, name,<br/>description, body, triggers, scope, confidence, cited event and episode ids)<br/>and a verdict per recalled card: helped, hurt, or irrelevant
      alt Valid response
        S->>S: recall RecordRecallOutcome(verdicts) (recall SEQ-054)
      else Error, timeout, or invalid JSON
        S->>DB: episode.replay_attempts + 1 · replay_state 'skipped' at extraction_max_attempts
        Note over S: Remaining REPLAY episodes stay pending when the provider itself failed
      end
    else allowance = 0 without explicit_remember
      Note over S: The episode stays pending for tomorrow's allowance
    end
    loop Each proposed operation
      S->>S: Validate the proposal
      Note over S: GUARD · kind and op allowed by the RULE-021 route, or retire of an active open-loop<br/>body of 1 to 2000 characters · at least one cited id inside this episode<br/>target card exists in a readable space · scope 'user' only for preference or lesson<br/>Near-duplicate of a native memory line (cosine ≥ native_dedupe_similarity) becomes noop
      S->>S: Decide with RULE-022 (invalid proposals become REJECT)
      S->>DB: Record the operation
      Note right of DB: INSERT · DATA-006 card_op<br/>WRITE · run_id, episode_id, op, kind, target_card_id, target_version,<br/>proposal, cited ids, evidence_trust, policy,<br/>state 'pending_review' (REVIEW or QUARANTINE), 'refused' (REJECT),<br/>or 'applying' (AUTO)
      opt policy AUTO
        S->>S: memory ApplyCardOp(op, expected_version = target_version) (memory SEQ-040)
        alt applied
          S->>DB: card_op.state = 'applied'
        else version conflict
          S->>DB: card_op.state = 'conflict' · the episode returns to pending for the next run
        else refused by memory
          S->>DB: card_op.state = 'refused', reason
        end
      end
    end
    S->>DB: episode.replay_state = 'replayed' unless it returned to pending
    S->>S: Charge the provider's reported tokens to the allowance and DATA-005 dreamer_tokens
  end
  S->>S: retention RetentionSweep(space) (retention SEQ-060)
  S->>S: recall MaintainIndex(space) (recall SEQ-055)
  S->>S: memory WriteChangelog(run) (memory SEQ-042)
  break Unrecoverable error (database unavailable)
    Note over S,DB: Committed operations stay applied · unprocessed episodes stay pending<br/>SEQ-030 records state 'failed' with the error, shown in kernle status
  end
```

### Current rationale

- Appraisal and judging happen before replay selection, so replay ranks
  episodes by their best available appraisal.
- The run pauses when the daily allowance is spent instead of degrading to
  cheaper extraction. Pending episodes keep their priority and resume the next
  day, while explicit remember requests still become cards without the Dreamer.
- Native memory is read only to avoid duplication. Kernle adds what native
  memory lacks rather than restating what the runtime already loads every
  session.
- A version conflict sends the episode back to pending rather than retrying
  at once. The card changed during the run, often through an owner edit, and
  the next run should consider the new content.
- Every proposed change is recorded in `card_op`, including refusals, so the
  changelog and review queue can explain everything sleep did or chose not to
  do.

## SEQ-032 Resolve a held change

The owner runs `kernle review` in an interactive terminal, or the daily
expiry pass runs inside the scheduler's retention run. Rust function
`sleep::review()` owns it.

**DCL:** 3 (product default)

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
    noteFontSize: 14
    noteFontFamily: Arial
  themeCSS: |
    .noteText { font-family: Arial !important; font-size: 14px !important; }
  themeVariables:
    noteBkgColor: '#242626'
    noteTextColor: '#e5edf5'
---
sequenceDiagram
  autonumber
  actor O as ACTOR-001 Owner
  participant C as ARCH-007 kernle CLI
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db

  alt Owner review
    O->>C: kernle review
    alt stdin or stdout is not an interactive terminal
      C-->>O: List held changes read-only · accept, edit, and reject are refused
    else Interactive terminal
      C->>D: List held changes for the requested spaces
      D->>DB: Read the queue
      Note right of DB: READ · DATA-006 card_op<br/>ACCESS · DATA-006 card_op [P1] card_op_review_queue_idx
      D-->>C: Changes with policy reason, appraisal, and cited evidence summary
      loop Each change shown (consolidation FLOW-003)
        O->>C: Accept, Edit, Reject, or Skip
        alt Accept a held delete (agent forget)
          C->>D: retention deletion steps of the unprotected-card branch (retention SEQ-061)
          D->>DB: card_op.state = 'accepted', decided_at
        else Accept
          C->>D: memory ApplyCardOp(proposal, expected_version = target_version) (memory SEQ-040)
          alt applied
            D->>DB: card_op.state = 'accepted', decided_at
          else version conflict
            D-->>C: Card changed since this proposal · shown again with the current card
          end
        else Edit
          C->>O: Open $EDITOR on the proposed card
          O-->>C: Saved text
          C->>D: memory ApplyCardOp(edited proposal, author owner) (memory SEQ-040)
          D->>DB: card_op.state = 'accepted', decided_at
        else Reject
          C->>D: Reject
          D->>DB: card_op.state = 'rejected', decided_at
        end
      end
    end
  else Daily expiry (inside the retention run, SEQ-030)
    D->>DB: Expire stale held changes
    Note right of DB: UPDATE · DATA-006 card_op state 'expired', decided_at<br/>ACCESS · DATA-006 card_op [P1]<br/>GUARD · created_at older than review_expiry_days<br/>NO WRITE · no card is changed
  end
```

### Current rationale

- Accepting or rejecting requires an interactive terminal. Agents run under
  the owner's account and can execute `kernle`, but their tool shells are not
  interactive. A quarantined change therefore cannot be approved by the same
  agent whose input produced it.
- Held changes expire rather than accumulate. A queue the owner never reads
  must not become a backlog that later applies stale memories in bulk.
- An edited acceptance is owner-authored. The owner's words become the card,
  and RULE-022 then protects it from automatic rewrites.
