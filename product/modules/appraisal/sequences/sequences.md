# Appraisal sequences

Appraisal turns events into episodes with an explainable appraisal and label
([DATA-004](../data/data-model.md)). Both processes run inside a sleep run
([consolidation SEQ-031](../../consolidation/sequences/sequences.md#seq-031-sleep-run))
while that run holds the space's maintenance lease. Rust module `appraise` owns
them.

## Public boundary

| Operation | Caller | Result and guarantees |
| --- | --- | --- |
| [AppraisePending](#seq-020-segment-and-appraise-new-events) | consolidation SEQ-031 | Every closed run of unsegmented events becomes exactly one episode with heuristic appraisal and label. Open episodes wait. Rerunning after a failure creates no duplicates. No network use |
| [JudgeEpisodes](#seq-021-judge-episodes) | consolidation SEQ-031 | Refines at most `judge_max_per_run` eligible episodes through the Dreamer, within the run's token allowance. A failed judgment keeps the heuristic appraisal. Never sends episodes of a runtime whose `dreamer_allowed` is false |

## SEQ-020 Segment and appraise new events

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
  participant E as ARCH-008 embedding model

  S->>DB: Load unsegmented events by session
  Note right of DB: READ · DATA-002 event<br/>ACCESS · DATA-002 event [P1] event_unsegmented_idx
  loop Each session with unsegmented events
    S->>S: Split the session's events into episodes
    Note over S: OWNER · appraise::segment()<br/>A task user_message starts a new episode<br/>A user_message is feedback, not a task, when it matches<br/>a correction, frustration, praise, or directive marker,<br/>or answers an assistant message ending in a question<br/>Feedback and interrupts stay in the current episode
    loop Each candidate episode
      alt Closed (a later task message, a session_end event, or no new event for session_idle_close_minutes)
        S->>E: Embed the summary
        Note over S: INPUT · summary = first 120 characters of the task message
        S->>S: Score with RULE-010 then label with RULE-011 (rules.md)
        Note over S: INPUT · nearest active cards from recall NearestCards (recall SEQ-055)<br/>baseline, recurrence, and novelty from DATA-004 episode rows of this space
        S->>DB: Store the episode and link its events in one transaction
        Note right of DB: INSERT · DATA-004 episode ON CONFLICT DO NOTHING<br/>CONSTRAINT · [U1] episode_session_first_seq_key<br/>WRITE · signals, arousal, valence, salience, label,<br/>peak_seq, evidence_first_seq, summary_vec,<br/>state 'appraised', replay_state 'pending',<br/>expires_at = now + episode_retention_days (retention settings)<br/>UPDATE · DATA-002 event.episode_id for seq first_seq to last_seq<br/>TRANSACTION · episode and event links commit together
      else Still open (the session may still send feedback)
        Note over S: NO WRITE · the events stay unsegmented for a later run
      end
    end
  end
  break Database error
    Note over S,DB: The failed episode's transaction rolls back and its events stay unsegmented<br/>Already committed episodes remain, and the next run continues
  end
```

### Current rationale

- An episode stays open until the session moves on. The person's reaction to
  a piece of work usually arrives in their next message, and appraising before
  it arrives would miss the strongest signal.
- Feedback messages belong to the episode they react to, not a new one. A
  correction is information about the work just done.
- Episode creation and event linking commit together, and `[U1]` makes a
  repeated run a no-op. A crash mid-run must not leave events in two episodes
  or none.

## SEQ-021 Judge episodes

Called by SEQ-031 after SEQ-020, with the run's remaining Dreamer token
allowance. Rust function `appraise::judge()` owns it.

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
  participant LM as ARCH-011 Dreamer provider

  S->>DB: Select episodes to judge
  Note right of DB: READ · DATA-004 episode<br/>KEY · state = 'appraised' and judge_attempts = 0<br/>GUARD · salience ≥ judge_min_salience, or social ≠ 0,<br/>or the episode contains an explicit_remember event<br/>GUARD · runtimes.{runtime}.dreamer_allowed is true (spaces config)<br/>ORDER · salience descending · LIMIT judge_max_per_run
  loop Each selected episode while the token allowance remains
    S->>DB: Load the evidence excerpt
    Note right of DB: READ · DATA-002 event<br/>ACCESS · DATA-002 event [I1] event_session_seq_idx<br/>KEY · session_id, seq from evidence_first_seq to last_seq,<br/>plus the task message at first_seq<br/>Excerpts are already scrubbed at capture (capture SEQ-011)
    S->>LM: Judge request (excerpt and the RULE-010 heuristic signals)
    Note right of LM: Asks for JSON: summary, expected, actual,<br/>surprise, outcome, social, directive,<br/>attribution (agent, user, or environment),<br/>open_loop, misleading_source_event_ids<br/>Each excerpt is capped at judge_max_input_tokens
    alt Valid JSON with every value in range
      S->>S: Rescore with RULE-010 and relabel with RULE-011 using the judged values
      S->>DB: Store the judged appraisal
      Note right of DB: UPDATE · DATA-004 episode<br/>WRITE · judged signals, arousal, valence, salience, label,<br/>summary, state 'judged', judge_attempts + 1<br/>KEY · id
    else Timeout, provider error, or invalid response
      S->>DB: Record the attempt only
      Note right of DB: UPDATE · DATA-004 episode.judge_attempts + 1<br/>PRESERVE · the heuristic appraisal and label stay in force
    end
    S->>S: Charge the provider's reported tokens to the run's allowance
  end
```

### Current rationale

- Only the top slice is judged, in salience order. Judging every episode
  would spend the owner's Dreamer budget on routine work, while the episodes
  that decide what is remembered get the best evidence.
- Each episode is judged at most once. A provider outage must not spend the
  budget retrying the same episode every night, and the heuristic appraisal is
  already a complete answer.
- The judge sees the evidence window, not the whole session. That bounds both
  cost and how much private content leaves the machine.
