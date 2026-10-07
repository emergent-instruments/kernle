# Capture user flows

Capture is invisible during ordinary agent work. Its only actor-visible flow is
asking Kernle to remember something explicitly.

## FLOW-008 Remember something explicitly

Both paths run [SEQ-013](../sequences/sequences.md#seq-013-remember-explicitly).

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 300
  themeVariables:
    clusterBkg: '#18222d'
    clusterBorder: '#344556'
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  A0(["ACTOR-001 tells the agent: remember this"]) --> A1
  subgraph S_AGENT["EXTERNAL · agent session (Claude Code, Codex, or OpenClaw)"]
    A1["Agent calls the remember tool"]
    A2["Agent receives: captured, becomes memory after this session's sleep"]
    A3["Agent receives: Kernle unavailable · agent says it could not save the memory"]
  end
  A1 -->|captured| A2
  A1 -->|Kernle not running or not initialized| A3
  A2 --> LATER(["After sleep the memory appears in kernle log,<br/>or in kernle review when policy holds it · consolidation FLOW-003"])

  O0(["ACTOR-001 runs kernle remember &quot;text&quot;"]) --> O1
  subgraph S_CLI["SURFACE · kernle remember"]
    O1["Show the new card's name, kind, space, and file path"]
    O2["Show refusal: empty text or over the card length limit"]
  end
  O0 -.->|text invalid| O2
```

## Current rationale

- The agent is told that memory forms after sleep, not that it is saved now.
  Agent requests pass through policy first, so the agent should not claim
  something is remembered before that happens.
- The owner's command shows the created file. Owners trust memory they can
  open and edit, and the card exists immediately for them.
