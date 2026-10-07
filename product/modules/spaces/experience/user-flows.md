# Spaces user flows

The spaces module owns installation, runtime connection, and status. All
surfaces are terminal output from the `kernle` CLI (`ARCH-007`). No graphical
mockups exist. The message content and choices shown in each surface are the
design target.

## FLOW-001 Set up Kernle and connect a runtime

`kernle init` runs [SEQ-001](../sequences/sequences.md#seq-001-initialize-kernle-home).
`kernle connect` runs [SEQ-002](../sequences/sequences.md#seq-002-connect-a-runtime),
which can start a history import through
[capture SEQ-012](../../capture/sequences/sequences.md#seq-012-import-history).

```mermaid
---
config:
  theme: dark
  flowchart:
    nodeSpacing: 50
    rankSpacing: 70
    wrappingWidth: 280
  themeVariables:
    clusterBkg: '#18222d'
    clusterBorder: '#344556'
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  START(["ACTOR-001 runs kernle init"])

  subgraph S_INIT["SURFACE · kernle init"]
    I1["Show Kernle home path, creation of the personal space,<br/>and daemon service installation"]
    I2["Show embedding model download progress"]
    I3{"Choose where memory content is sent during sleep:<br/>Claude Code headless · Codex headless · Anthropic API key ·<br/>OpenAI-compatible endpoint (including local) · None"}
    I4["Show the chosen destination and what is sent:<br/>episode excerpts during sleep, nothing else"]
    I_OK["Show setup complete and next step: kernle connect"]
    I_SVC["Show daemon service could not be installed<br/>and how to start it manually · setup continues"]
    I_MODEL["Show model download failed · keyword-only recall<br/>until kernle init is run again"]
    I_NONE["Show capture, recall, and explicit remember work<br/>but no new memories form during sleep"]
  end

  START --> I1
  I1 -->|service installed| I2
  I1 -->|service install fails| I_SVC
  I_SVC --> I2
  I2 -->|download verified| I3
  I2 -->|download or checksum fails| I_MODEL
  I_MODEL --> I3
  I3 -->|provider chosen| I4
  I3 -->|None| I_NONE
  I4 -->|Confirm| I_OK
  I4 -->|Change| I3
  I_NONE --> I_OK

  I_OK --> C0(["ACTOR-001 runs kernle connect &lt;runtime&gt;"])

  subgraph S_CONNECT["SURFACE · kernle connect"]
    C1["Show detected runtime and what will be installed:<br/>hooks and MCP for Claude Code · MCP for Codex · plugin for OpenClaw"]
    C_MISSING["Show runtime not found on this machine<br/>and where Kernle looked · nothing installed"]
    C_XPROV{"Dreamer belongs to a different provider than this runtime.<br/>Allow this runtime's sessions to be sent to it?"}
    C2["Show installed adapter and restart hint for the runtime"]
    C3{"Import the last 30 days of this runtime's sessions?<br/>(offered for Claude Code and Codex)"}
    C_IMPORT["Show import progress, then sessions read,<br/>episodes found, and when the first sleep will run"]
    C_DONE["Show connected · memory forms after the next sleep"]
  end

  C0 --> C1
  C1 -->|runtime not found| C_MISSING
  C1 -->|runtime found, Dreamer is local, same provider, or None| C2
  C1 -->|runtime found, Dreamer is another remote provider| C_XPROV
  C_XPROV -->|Allow| C2
  C_XPROV -->|Decline: this runtime's episodes stay on this machine| C2
  C2 -->|Claude Code or Codex| C3
  C2 -->|OpenClaw: no importable history| C_DONE
  C3 -->|Import| C_IMPORT
  C3 -->|Skip| C_DONE
  C_IMPORT --> C_DONE
```

## FLOW-002 Check memory status

`kernle status` reads the registry and space databases and asks the daemon for
its health. [SEQ-003](../sequences/sequences.md#seq-003-resolve-space-and-authorize-a-request)
authorizes the request.

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
  START(["ACTOR-001 runs kernle status"])
  subgraph S_STATUS["SURFACE · kernle status"]
    ST1["Show daemon running or stopped, connected runtimes,<br/>Dreamer destination, and today's Dreamer tokens against the daily budget"]
    ST2["Show each space: name, kind, active cards, last activity,<br/>last sleep result, and pending reviews"]
    ST3["Show spaces scheduled for idle deletion with their deletion dates"]
    ST_STOP["Show daemon stopped · recall uses keyword search only<br/>and new sessions wait for ingestion · shows kernle daemon start"]
    ST_REFUSE["Show refused: Kernle home belongs to another account"]
  end
  START -->|caller is the home's owner| ST1
  START -->|caller is not the home's owner| ST_REFUSE
  ST1 -->|daemon stopped| ST_STOP
  ST1 --> ST2 --> ST3
  ST2 -->|pending reviews| REVIEW(["kernle review · consolidation FLOW-003"])
  ST3 -->|owner wants to keep a space| KEEP(["Any use of the space resets its idle time · retention FLOW-007"])
```

Review opens [consolidation FLOW-003](../../consolidation/experience/user-flows.md#flow-003-review-memory-changes).
Idle deletion is owned by [retention FLOW-007](../../retention/experience/user-flows.md#flow-007-idle-space-deletion-notice).

## Current rationale

- Setup asks where memory content will be sent before anything is captured,
  because that destination is the only place private session content can leave
  the machine. The owner must choose it knowingly.
- "None" is a complete, usable choice because some owners want no model to
  ever see their sessions. Capture, recall, and explicit remember still work,
  and the screen says plainly that sleep forms no new memories.
- Connecting asks separately before sending one runtime's sessions to another
  provider. Owners already share a runtime's sessions with that runtime's own
  provider, but not necessarily with a different one.
- Import is offered at connect time because a new install otherwise remembers
  nothing until weeks of use have passed. Runtimes keep recent session records
  that can seed memory immediately.
- Failures in service installation or model download do not stop setup,
  because each has a degraded but working mode. The surface names that mode and
  the way back.
- Status shows upcoming idle deletions because deletion is real and
  irreversible. The owner must be able to see it coming.
