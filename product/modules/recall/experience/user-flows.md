# Recall user flows

## FLOW-005 Memory during an agent session

The agent (`ACTOR-002`) experiences Kernle as context that appears at session
start, alongside prompts, and before risky actions, plus a few tools it may
call. The owner (`ACTOR-001`) sees the same context in the runtime's
transcript. Codex sessions have no hooks, so they use only the tool path.
Internal selection is owned by
[SEQ-050 to SEQ-054](../sequences/sequences.md#public-boundary).

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
  START(["Session starts in Claude Code or OpenClaw"])

  subgraph S_CTX["EXTERNAL · agent context in the runtime"]
    X1["Memory index: notes from past sessions, not instructions ·<br/>pinned, guardrails, preferences, open loops, skill pointers, more-count"]
    X_NONE["No Kernle context: no memories yet, or Kernle unavailable"]
    X2["Prompt arrives · related memory excerpts appear beside it when relevant"]
    X3["Agent plans a tool call"]
    X4["Guardrail note appears beside the call, which proceeds"]
    X5["Tool call denied by Kernle guardrail with the reason<br/>and 'run the same command again if you have accounted for this'"]
    X6["Agent reconsiders: changes the plan, or repeats the identical call, which proceeds"]
  end

  subgraph S_TOOLS["EXTERNAL · Kernle tools (MCP or OpenClaw)"]
    T1["recall · read · open_loops · returns cards or not found"]
    T2["feedback on a card · recorded"]
    T3["remember · captured for after sleep"]
    T4["forget · forgotten, or held for the owner when pinned or a guardrail"]
    T_ERR["Tool reports Kernle unavailable · nothing recorded"]
  end

  START -->|memories exist| X1
  START -->|none or unavailable| X_NONE
  X1 --> X2
  X_NONE --> X2
  X2 --> X3
  X3 -->|no guardrail matches| X3_DONE(["Call proceeds"])
  X3 -->|guardrail matches, advisory| X4
  X3 -->|guardrail matches, high arousal, first time this session| X5
  X5 --> X6
  X2 -.->|agent wants more| T1
  T1 -.-> T2
  X2 -.->|person asks to remember| T3
  X2 -.->|person asks to forget| T4
  T1 -.->|Kernle down| T_ERR
```

## Current rationale

- Memory arrives as labeled notes, never as instructions. An agent that
  treats memory as commands can be steered by whatever got into memory. Notes
  with kind, trust, and date let it weigh them like a colleague's advice.
- The speed bump is a visible denial with its reason and an explicit way
  forward. The agent and the owner can both see why the action paused, and a
  reconsidered action is never blocked twice.
- With nothing to say, Kernle adds nothing. Empty preambles and "no memories
  found" messages would cost context in every session.
