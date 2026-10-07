# Retention user flows

## FLOW-006 Forget a memory

The owner path runs
[SEQ-061](../sequences/sequences.md#seq-061-forget-on-request). The agent path
is shown in [recall FLOW-005](../../recall/experience/user-flows.md#flow-005-memory-during-an-agent-session).

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
  START(["ACTOR-001 runs kernle forget &lt;name, id, or text&gt;"])
  subgraph S_FORGET["SURFACE · kernle forget"]
    F_PICK{"Several cards match · pick one"}
    F_PREVIEW["Show what will be deleted: the card, N versions, predecessors,<br/>N episodes and N events used only by it, and the projected skill<br/>List other memories that rely on it, as optional additions<br/>Note: Claude Code and Codex keep their own transcripts of those sessions"]
    F_CONFIRM{"Delete permanently? select optional cards · Confirm · Cancel"}
    F_DONE["Show counts removed · the change appears in kernle log without the memory's name"]
    F_CANCEL["Show nothing deleted"]
    F_NOTFOUND["Show no matching memory"]
    F_NOTTY["Show refused: forget needs an interactive terminal"]
  end
  START -->|one match| F_PREVIEW
  START -->|several matches| F_PICK --> F_PREVIEW
  START -->|no match| F_NOTFOUND
  START -->|run without an interactive terminal| F_NOTTY
  F_PREVIEW --> F_CONFIRM
  F_CONFIRM -->|Confirm| F_DONE
  F_CONFIRM -->|Cancel| F_CANCEL
```

## FLOW-007 Idle space deletion notice

Deletion runs in [SEQ-062](../sequences/sequences.md#seq-062-idle-space-deletion).
The notice appears in [spaces FLOW-002](../../spaces/experience/user-flows.md#flow-002-check-memory-status).

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
  IDLE(["A space has had no agent activity for 11 months"])
  subgraph S_STATUS["SURFACE · kernle status"]
    N1["Show: &lt;space&gt; idle since &lt;date&gt;, deleted on &lt;date&gt; unless used"]
  end
  IDLE --> N1
  N1 -->|owner or agent works in that repository again| RESET(["Idle time resets · notice disappears"])
  N1 -->|no use before the date| GONE(["Space deleted · it no longer appears in status"])
```

## Current rationale

- The forget preview lists the evidence as well as the card. "Forget that"
  usually means the underlying conversation too, and the owner should choose
  knowingly.
- The preview reminds the owner that runtimes keep their own transcripts.
  Kernle can only promise deletion of what it holds, and the owner may need to
  delete those transcripts as well.
- Idle deletion has a notice but no "keep" button. Using the space is the way
  to keep it, which matches the rule that memory survives through use.
