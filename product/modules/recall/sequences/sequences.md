# Recall sequences

Recall brings memory back into agent context and records how it was used. It
owns [DATA-012 and DATA-013](../data/data-model.md), the
[rules](../behavior/rules.md), and the agent tool contract
([mcp-tools.yaml](../contracts/mcp-tools.yaml)). Rust module `recall` owns
these processes. The hook processes run inside capture's
[SEQ-010 deadline](../../capture/sequences/sequences.md#seq-010-handle-a-runtime-hook-event).

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [SessionIndex](#seq-050-session-index) | capture SEQ-010 | Index text within `tier0_budget_tokens`, or empty. Never fails the hook |
| [CueInject](#seq-051-cue-injection) | capture SEQ-010 | Zero to `cue_max_cards` excerpts within `cue_budget_tokens`. Cards already shown this session are not repeated |
| [GutCheck](#seq-052-gut-check) | capture SEQ-010 | `none`, `advise` with text, or `speed_bump` with a reason. At most one speed bump per card and action per session |
| [Recall, Read, Open loops](#seq-053-deliberate-recall) | MCP and OpenClaw tools, `kernle recall` | Ranked cards, one full card with its sources, or open loops. Never another project's space for agent requests |
| [Feedback and RecordRecallOutcome](#seq-054-feedback-and-recall-outcomes) | agent `feedback` tool, consolidation SEQ-031 | Updates usage counters and `last_used_at` |
| [MaintainIndex and NearestCards](#seq-055-maintain-the-search-index) | consolidation SEQ-031, retention SEQ-060 and SEQ-061, spaces SEQ-004, appraisal SEQ-020 | The index reflects every ledger change up to its cursor. Deleted cards leave the index before MaintainIndex returns |

## SEQ-050 Session index

**DCL override:** 4 — runs inside the agent's session-start hook.

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
  participant A as Adapter (ARCH-007 hook or ARCH-006 plugin)
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db files

  A->>D: SessionIndex(readable spaces, session_id, runtime)
  alt Daemon answering
    D->>D: Select cards with RULE-031 from cached card rows
    D->>DB: Log the inclusion
    Note right of DB: INSERT · DATA-012 recall_log tier 'index', one row per card<br/>UPDATE · DATA-007 card.recalled_count + 1<br/>NO WRITE · last_used_at (inclusion alone is not use)
    D-->>A: Index text starting with kernle_injection_marker
  else Daemon not answering (direct mode)
    A->>DB: Open the space databases read-only and select with RULE-031
    Note right of DB: READ · DATA-007 card (status, pinned, kind, strength, salience)<br/>NO WRITE · nothing is logged in direct mode
    DB-->>A: Index text
  end
  Note over A: Empty when there are no active cards<br/>Output goes to SessionStart additionalContext or OpenClaw prependContext
```

### Current rationale

- Inclusion in the index does not count as use. Otherwise every card in the
  index would refresh its own idle time each session and never expire,
  whether or not it ever helped.
- Direct mode reads without writing, so recall still works while the daemon
  is stopped, without a second writer appearing.

## SEQ-051 Cue injection

**DCL override:** 4 — runs inside the agent's prompt-submit hook.

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
  participant A as Adapter (ARCH-007 hook or ARCH-006 plugin)
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db

  A->>D: CueInject(readable spaces, session_id, prompt)
  Note right of D: INPUT · prompt from UserPromptSubmit or before_prompt_build
  alt Prompt shorter than cue_min_prompt_chars, or a slash command
    D-->>A: Nothing
  else
    D->>D: Rank with RULE-030 (minimum cue_min_relevance)
    D->>DB: Exclude cards already shown in this session
    Note right of DB: READ · DATA-012 recall_log<br/>ACCESS · DATA-012 recall_log [I1] recall_log_session_idx<br/>KEY · session_id
    D->>D: Keep the top cue_max_cards that fit cue_budget_tokens
    Note over D: Each excerpt: kind, trust, date, name, and the body's first 300 characters
    alt No card qualifies
      D-->>A: Nothing
    else Cards selected
      D->>DB: Log the injection
      Note right of DB: INSERT · DATA-012 recall_log tier 'cue'<br/>UPDATE · DATA-007 card.recalled_count + 1
      D-->>A: Excerpts starting with kernle_injection_marker
    end
  end
  Note over A,D: Daemon not answering: lexical-only RULE-030 over read-only databases,<br/>without the shown-this-session exclusion and without logging
```

### Current rationale

- Recall is cue-driven: the person's actual request is the best cue, so
  injection waits for the prompt rather than guessing at session start.
- Cards already shown in the session are excluded. Repeating a memory the
  agent has in context only spends tokens.

## SEQ-052 Gut check

**DCL override:** 4 — runs inside the agent's pre-tool hook on every tool call.

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
  participant A as Adapter (ARCH-007 hook or ARCH-006 plugin)
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db

  A->>D: GutCheck(readable spaces, session_id, tool_name, tool_input)
  D->>DB: Read this session's gut-check history
  Note right of DB: READ · DATA-012 recall_log<br/>ACCESS · DATA-012 recall_log [I1] recall_log_session_idx<br/>KEY · session_id, tier in ('gut_advise', 'speed_bump')
  D->>D: Evaluate each active guardrail with RULE-032
  alt Any SPEED_BUMP
    D->>DB: Record the speed bump and the use
    Note right of DB: INSERT · DATA-012 recall_log tier 'speed_bump', action_key<br/>UPDATE · DATA-007 card.last_used_at = now, recalled_count + 1
    D-->>A: speed_bump · reason: the guardrail text and<br/>"If you have accounted for this, run the same command again."
  else Any ADVISE
    D->>DB: Record the advice and the use
    Note right of DB: INSERT · DATA-012 recall_log tier 'gut_advise'<br/>UPDATE · DATA-007 card.last_used_at = now, recalled_count + 1
    D-->>A: advise · up to gut_check_max_warnings guardrails as context
  else All NONE
    D-->>A: none
  end
  Note over A,D: Daemon not answering: trigger-list matching only over read-only databases,<br/>never SPEED_BUMP, nothing logged
```

### Current rationale

- A trigger match counts as use. A guardrail exists to fire at the risky
  moment, so firing keeps it alive, while one whose situation never recurs
  can expire on schedule.

## SEQ-053 Deliberate recall

Agent tools `recall`, `read`, and `open_loops` (MCP or OpenClaw), and owner
command `kernle recall <query>`.

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
  participant Q as Agent tool or owner CLI
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db and card files

  Q->>D: recall(query, limit, include_dormant) · read(name or id) · open_loops()
  D->>D: spaces ResolveSpace (spaces SEQ-003)
  alt recall
    D->>D: Rank with RULE-030 (minimum search_min_relevance)
    opt Requester is an agent
      D->>DB: INSERT · DATA-012 recall_log tier 'search' · UPDATE · DATA-007 card.recalled_count + 1
    end
    D-->>Q: Up to limit cards: name, kind, trust, date, description, score
  else read
    D->>DB: Load the card and its sources
    Note right of DB: READ · DATA-007 card · KEY · [U1] card_name_key or id<br/>READ · DATA-011 file content · DATA-009 card_source with episode summaries<br/>GUARD · the card is in one of the requester's readable spaces, otherwise not found
    opt Requester is an agent
      D->>DB: INSERT · DATA-012 recall_log tier 'read' · UPDATE · DATA-007 card.last_used_at = now, recalled_count + 1
    end
    D-->>Q: Full card, appraisal, and dated source summaries · or not found
  else open_loops
    D->>DB: READ · DATA-007 card · KEY · kind 'open-loop', status 'active', project space
    D-->>Q: Open loops, newest first
  end
```

### Current rationale

- An agent's read counts as use, while an owner browsing with `kernle recall`
  does not. Owner curiosity should not keep memories alive that the agents
  never need.
- A card outside the readable spaces returns "not found" rather than
  "forbidden", so its existence is not disclosed.

## SEQ-054 Feedback and recall outcomes

Agent tool `feedback(card, verdict, note?)`, and RecordRecallOutcome from
consolidation SEQ-031 with Dreamer verdicts.

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
  participant Q as Agent feedback tool or sleep run
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db

  Q->>D: feedback(card, helped, wrong, or outdated) · RecordRecallOutcome(card verdicts for an episode)
  alt helped (feedback or verdict)
    D->>DB: UPDATE · DATA-007 card.helped_count + 1, last_used_at = now
  else wrong or outdated (feedback), or hurt (verdict)
    D->>DB: UPDATE · DATA-007 card.hurt_count + 1
    Note right of DB: RULE-030 then ranks the card lower, and excludes it from<br/>automatic recall once hurt_count - helped_count reaches hurt_exclusion_margin
  else irrelevant (verdict)
    Note over D: No counter change
  end
  D->>DB: Record the outcome
  Note right of DB: UPDATE · DATA-012 recall_log.outcome for the matching rows,<br/>or INSERT a tier 'feedback' row for agent feedback
  D-->>Q: Recorded · or not found when the card is outside the readable spaces
```

### Current rationale

- Helpfulness, not exposure, keeps a memory alive and ranks it higher. This
  is reconsolidation: each recall's outcome updates the memory that was
  recalled.
- Wrong or outdated feedback never edits or deletes the card. The feedback
  may itself come from a manipulated agent, so it only lowers ranking.
  Deletion stays with retention and the owner.

## SEQ-055 Maintain the search index

Triggered by the daemon every `index_poll_seconds`, at daemon start (spaces
SEQ-004), at the end of each sleep run, and synchronously after deletions
(retention SEQ-060 and SEQ-061).

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
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db
  participant E as ARCH-008 embedding model

  D->>DB: Read index_state
  Note right of DB: READ · DATA-013 index_state (ledger_cursor, model_id)
  alt index_state missing, or model_id differs from embedding.model_id
    D->>DB: Rebuild card_fts and card_vector for every non-superseded card, then set the cursor to the latest ledger id
  else
    D->>DB: Read ledger entries after the cursor
    Note right of DB: READ · DATA-010 card_ledger · KEY · id greater than ledger_cursor (PK order)
    loop Each affected card
      alt change = deleted, or the card is superseded
        D->>DB: DELETE · DATA-013 card_fts and card_vector rows for the card
      else
        D->>DB: Upsert the card_fts row from the current file content
        opt embedding.available
          D->>E: Embed name, description, and body
          D->>DB: Upsert the card_vector row (vector, model_id, version)
        end
      end
    end
    D->>DB: UPDATE · DATA-013 index_state.ledger_cursor = last processed ledger id
  end
  Note over D: NearestCards(text or vector, spaces, k) is answered from the in-memory<br/>copy of card_vector by brute-force cosine over active cards
```

### Current rationale

- The index follows the ledger instead of being called by every writer. Owner
  edits, sleep, review, and retention all produce ledger entries, so none of
  them can forget to update the index.
- Deletions run synchronously before the deleting process returns, because
  the index holds card text and must not keep deleted content after the
  owner is told it is gone.
- Brute-force cosine is enough for a personal store of thousands of cards. An
  approximate index would add a dependency and nondeterminism without a
  latency need.
