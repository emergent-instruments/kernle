# Capture sequences

Capture turns agent sessions into private, scrubbed events
([DATA-002, DATA-003](../data/data-model.md)) without agent effort. It owns the
runtime adapters' entry point, ingestion of session records, history import,
and explicit remember. Rust module `capture` owns these processes. Runtime
mappings are in [runtime-events.yaml](../contracts/runtime-events.yaml) and
parameters in [settings.yaml](../contracts/settings.yaml).

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [HandleHookEvent](#seq-010-handle-a-runtime-hook-event) | Claude Code via `ARCH-005`, OpenClaw via `ARCH-006` | Returns within `hook_deadline_ms`. Never blocks or alters an agent action except the Kernle-home write guard and a recall speed bump. Every failure yields an empty, non-blocking response |
| [IngestSource](#seq-011-ingest-session-records) | SEQ-010, daemon catch-up (spaces SEQ-004), the Codex watcher, SEQ-012 | Idempotent. Each source position becomes at most one event. A failure leaves the cursor unchanged so a later trigger resumes. Scrubbed before storage |
| [ImportHistory](#seq-012-import-history) | spaces SEQ-002, `kernle import` | Same guarantees as IngestSource across many files. Returns counts and the next sleep time |
| [Remember](#seq-013-remember-explicitly) | MCP `remember`, OpenClaw `remember` tool, `kernle remember` | Agent requests are captured as events for the next sleep. Owner requests create a card immediately |

## SEQ-010 Handle a runtime hook event

Entry for every Claude Code hook and OpenClaw typed hook. Rust function
`capture::hook::handle()` owns dispatch. The shared deadline belongs to this
handler.

**DCL override:** 4 — runs inside the agent's critical path on every prompt and
tool call, so it must answer within the deadline and fail open without
supervision.

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
  participant RT as Runtime (ARCH-001 or ARCH-003)
  participant A as Adapter (ARCH-007 kernle hook or ARCH-006 plugin)
  participant D as ARCH-008 kernled

  RT->>A: Hook event with session_id, cwd, transcript_path or turn record, prompt or tool call
  Note right of A: INPUT · event name and payload from the runtime (runtime-events.yaml)<br/>deadline = hook_deadline_ms from capture settings, started now
  A->>D: spaces ResolveSpace(role agent_hook or openclaw_plugin, cwd, runtime, session_id)
  alt Refused, home not initialized, or daemon unreachable for a write event
    A-->>RT: Empty response · exit 0 · agent continues unchanged
  else Space resolved
    alt Session start (Claude Code SessionStart, or the first OpenClaw before_prompt_build of a session)
      A->>D: recall SessionIndex(spaces, session) (recall SEQ-050)
      D-->>A: index text
      A-->>RT: additionalContext (Claude Code) or prependContext (OpenClaw)
    else Prompt submitted (UserPromptSubmit, or each OpenClaw before_prompt_build)
      A->>D: recall CueInject(spaces, session, prompt) (recall SEQ-051)
      D-->>A: zero or more memory excerpts
      A-->>RT: additionalContext, or empty when there are none
    else Before a tool call (PreToolUse, OpenClaw before_tool_call)
      alt Tool writes, edits, moves, or deletes a path inside ~/.kernle
        A-->>RT: Deny · reason: Kernle memory files are owner-edited. Use the remember or forget tool instead.
        Note over A: GUARD · path from tool_input file_path, or Bash command path arguments<br/>NO WRITE · the file is not touched
      else Any other tool call
        A->>D: recall GutCheck(spaces, session, tool_name, tool_input) (recall SEQ-052)
        D-->>A: none, advise with text, or speed_bump with reason
        A-->>RT: nothing, additionalContext, or deny with reason (as returned)
      end
    else Activity to ingest (Stop as async hook, PreCompact, SessionEnd, OpenClaw agent_end or session_end)
      alt Daemon reachable
        A->>D: Enqueue IngestSource(transcript_path, or the forwarded OpenClaw turn record)
        Note right of D: Continues in SEQ-011 after the response<br/>Repeated enqueues of one source coalesce
      else Daemon unreachable and runtime is OpenClaw
        A->>A: Append the turn record to ~/.kernle/spool/openclaw.jsonl (mode 0600)
        Note over A: The daemon ingests the spool at its next start (spaces SEQ-004)
      else Daemon unreachable and runtime is Claude Code
        Note over A: NO WRITE · daemon catch-up re-reads the transcript later
      end
      A-->>RT: Empty response
    end
  end
  break Deadline passes or any step fails
    A-->>RT: Empty response · exit 0 · failure appended to ~/.kernle/logs/hook.log
  end
```

### Current rationale

- Every failure path returns an empty, non-blocking response. A memory system
  that can stall or break the agent's own work costs more than the memory is
  worth.
- Turn-end ingestion only enqueues work. Claude Code gives all `SessionEnd`
  hooks 1.5 seconds, and reading a transcript inside the agent's turn would add
  visible latency.
- Writes into `~/.kernle` are denied from agent tools. Otherwise an agent, or
  instructions injected into it, could rewrite memory files directly. That
  would bypass provenance, policy, and the watcher's assumption that file
  edits come from the owner.
- OpenClaw turns are spooled when the daemon is down. Kernle cannot re-read
  OpenClaw session storage later the way it re-reads Claude Code transcripts.

## SEQ-011 Ingest session records

Triggered by SEQ-010 enqueues, the daemon's catch-up and spool scan
(spaces SEQ-004), the Codex watcher (a Codex session file unchanged for
`codex_idle_ingest_minutes`), and SEQ-012. Rust function
`capture::ingest::ingest_source()` owns it.

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
  participant W as ARCH-008 ingestion worker
  participant F as ARCH-004 session records or spool
  participant DB as ARCH-009 space kernle.db

  W->>F: Read the source header
  Note right of F: INPUT · cwd and session_id from the first record<br/>(transcript, Codex session meta, or forwarded turn)
  W->>W: spaces ResolveSpace(role capture, cwd, runtime) (spaces SEQ-003)
  W->>DB: Read the cursor for this source
  Note right of DB: READ · DATA-003 ingest_cursor<br/>KEY · source_key = runtime + ':' + absolute path or OpenClaw session id
  alt File identity changed (device or inode differs, or size below byte_offset)
    Note over W: Start again from byte_offset 0<br/>[U1] makes already-stored positions no-ops
  end
  W->>F: Read complete records after the cursor byte_offset
  Note right of F: GUARD · stop before a trailing partial line still being written
  loop Each complete record
    W->>W: Classify the record with RULE-001 (rules.md)
    alt RULE-001 returns DROP
      Note over W: Kernle's own injected context and runtime bookkeeping<br/>NO WRITE · never stored, which prevents memory reinforcing itself
    else RULE-001 returns an event kind, actor, and trust
      W->>W: Scrub the payload
      Note over W: Replace matches of scrub_patterns with [REDACTED:class]<br/>Drop binary and image content, keeping a placeholder<br/>Truncate text over max_event_chars, keeping head and tail
    end
  end
  W->>DB: Store the events and advance the cursor in one transaction
  Note right of DB: INSERT · DATA-002 event ON CONFLICT DO NOTHING<br/>CONSTRAINT · [U1] event_source_position_key<br/>WRITE · seq = next per-session sequence<br/>ingested_at = now<br/>expires_at = now + event_retention_days (retention settings)<br/>episode_id NULL (pending appraisal)<br/>UPDATE · DATA-003 ingest_cursor byte_offset, next_pos, file_dev, file_ino<br/>TRANSACTION · events and cursor commit together
  alt Commit succeeds
    W->>W: consolidation RequestMaintenance(space, trigger) (consolidation SEQ-030)
    Note over W: trigger = volume when the space's unsegmented events (DATA-002 [P1])<br/>number at least volume_threshold_events, else activity
  else Database error
    Note over W,DB: Roll back · cursor unchanged · the next trigger resumes from the same offset
  end
  opt A record cannot be parsed
    Note over W: Skip the record, count it in the run's parse_errors<br/>(shown in kernle status) and continue
  end
```

### Current rationale

- Ingestion reads the runtime's own records rather than hook payloads, because
  the records hold every message, tool call, and result in order. A missed
  hook then only delays memory.
- Events and cursor commit together, and `[U1]` makes every position
  idempotent. Hooks, catch-up, and import can all trigger the same file, and
  each record must still become exactly one event.
- Scrubbing happens before anything is stored. A secret that once reaches the
  database also reaches episodes, Dreamer requests, and cards.
- Kernle's own injected context is dropped. Otherwise each recall would be
  re-captured as experience and strengthen itself in a loop.

## SEQ-012 Import history

Runs from `kernle connect` (spaces SEQ-002) or `kernle import <runtime> [--since]`.
Import is available for Claude Code and Codex, which keep session records on
disk. OpenClaw history is not importable. Rust function
`capture::import::import_history()` owns it.

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
  participant F as ARCH-004 session records

  O->>C: Import (from connect) or kernle import claude-code or codex
  C->>D: ImportHistory(runtime, since)
  Note right of D: INPUT · since from --since, default now - 30 days
  alt runtime is openclaw
    D-->>C: Not available · OpenClaw history cannot be imported
  else claude-code or codex
    D->>F: List session record files modified at or after since
    Note right of F: claude-code · ~/.claude/projects/*/*.jsonl<br/>codex · ~/.codex/sessions/**/*.jsonl
    loop Each file, oldest first
      D->>D: IngestSource(file) (SEQ-011)
      D-->>C: Progress: files done of total
    end
    D->>D: consolidation RequestMaintenance(each touched space, trigger import) (consolidation SEQ-030)
    D-->>C: sessions read, events stored, parse errors, next sleep time
    C-->>O: Show the import summary (FLOW-001)
  end
```

### Current rationale

- Import reuses ingestion. Re-importing, or importing a session that hooks
  already captured, then creates nothing twice.
- The default window is 30 days, the same as event retention. Imported events
  older than that would expire on arrival.

## SEQ-013 Remember explicitly

The agent calls `remember` (MCP or OpenClaw tool) during a session. The owner
runs `kernle remember "<text>" [--kind <kind>] [--personal]`. Rust function
`capture::remember()` owns it.

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
  participant Q as Agent tool call or owner CLI
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db

  Q->>D: Remember(text, kind?, scope?)
  D->>D: spaces ResolveSpace (spaces SEQ-003)
  alt Requester is the agent (MCP or OpenClaw tool)
    D->>DB: Store an explicit_remember event
    Note right of DB: INSERT · DATA-002 event<br/>WRITE · kind 'explicit_remember', actor 'agent',<br/>trust 'agent_inferred', session_id from the request,<br/>payload = text and requested kind<br/>CONSTRAINT · [U1] source_key 'remember:' + request id
    D-->>Q: Captured · becomes memory after this session's sleep
    Note over D: Consolidation SEQ-031 always files explicit_remember events,<br/>and its policy decides whether the card is applied or reviewed
  else Requester is the owner (kernle remember)
    D->>DB: Store an explicit_remember event with actor 'principal', trust 'principal'
    D->>D: memory ApplyCardOp(add, kind or 'fact', scope, body = text, sources = that event) (memory SEQ-040)
    alt Applied
      D-->>Q: Show the new card's name and file path
    else Refused (empty or over the card length limit)
      D-->>Q: Show the refusal reason · the event remains for the next sleep
    end
  end
```

### Current rationale

- An agent's remember request waits for sleep instead of becoming a card at
  once. Instructions injected into an agent, for example from a web page,
  could otherwise plant a guardrail or preference that shapes the very next
  session. Sleep's policy checks what supports the request first.
- The owner's request becomes a card immediately because the owner is the
  principal. There is nothing to verify, and the owner expects to see the
  result.
