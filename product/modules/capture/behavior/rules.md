# Capture rules

## RULE-001 Classify a session record

[SEQ-011](../sequences/sequences.md#seq-011-ingest-session-records) applies this
rule to every complete record. The result decides whether the record is stored,
as which event kind, and with which actor and trust tier. Trust tiers later
decide which memories may apply automatically
([consolidation RULE-022](../../consolidation/behavior/rules.md#rule-022-change-policy)).

Inputs come from the parsed record: its role and content block type, its tool
name and the tool's category in [settings.yaml](../contracts/settings.yaml), and
Kernle's injection marker.

```mermaid
---
config:
  theme: dark
  flowchart:
    wrappingWidth: 300
  themeVariables:
    edgeLabelBackground: '#1f2c38'
---
flowchart TD
  IN["RULE-001 input · one parsed record<br/>role and block type from the record<br/>tool name from the tool_use or tool_result block<br/>tool category from capture settings tool_categories<br/>marker: text starts with kernle_injection_marker,<br/>or record is a Kernle hook attachment or Kernle MCP tool call or result"]
  K{"Kernle-originated?"}
  BK{"Runtime bookkeeping?<br/>file snapshots, progress ticks,<br/>metadata-only records"}
  U{"User-role text typed by the person?<br/>not a tool result, not system-injected"}
  INT{"User-role interrupt marker?"}
  AS{"Assistant text?"}
  TH{"Assistant thinking block?"}
  TU{"Assistant tool_use?"}
  TR{"Tool result?"}
  CMP{"Compaction summary?"}
  SES{"Session start or end metadata?"}

  DROP["DROP · not stored"]
  E_USER["user_message · actor principal · trust principal"]
  E_INT["interrupt · actor principal · trust principal"]
  E_AST["assistant_message · actor agent · trust agent_inferred"]
  E_CALL["tool_call · actor agent · trust first_party<br/>payload: tool name and arguments"]
  E_STATUS["tool_status · actor tool · trust first_party<br/>payload: ok or error, exit code, duration,<br/>test passed and failed counts when parseable"]
  WR{"Tool category = local_write?<br/>edit or write confirmations"}
  E_OUT_FP["tool_output · actor tool · trust first_party"]
  E_OUT_TP["tool_output · actor external · trust third_party<br/>file reads, shell output, web, browser, MCP results"]
  E_CMP["compaction · actor agent · trust agent_inferred"]
  E_SES["session_start or session_end · actor tool · trust first_party"]
  OTHER["DROP · unknown record types are counted as parse_errors"]

  IN --> K
  K -->|Yes| DROP
  K -->|No| BK
  BK -->|Yes| DROP
  BK -->|No| U
  U -->|Yes| E_USER
  U -->|No| INT
  INT -->|Yes| E_INT
  INT -->|No| AS
  AS -->|Yes| E_AST
  AS -->|No| TH
  TH -->|Yes| DROP
  TH -->|No| TU
  TU -->|Yes| E_CALL
  TU -->|No| TR
  TR -->|"Yes · emits two events"| E_STATUS
  TR -->|"Yes · emits two events"| WR
  WR -->|Yes| E_OUT_FP
  WR -->|No| E_OUT_TP
  TR -->|No| CMP
  CMP -->|Yes| E_CMP
  CMP -->|No| SES
  SES -->|Yes| E_SES
  SES -->|No| OTHER
```

### Current rationale

- A tool result is split into a first-party status and a separately trusted
  output. Whether a command succeeded is something the agent observed itself.
  What a file, web page, or command printed may have been written by anyone.
  Lessons grounded in success or failure can then apply automatically, while
  claims that rest only on outside text cannot.
- Shell output is third-party because commands like `cat` print content of
  unknown authorship. Instructions planted in a repository file must not be
  able to pass as first-party evidence.
- Thinking blocks are dropped. They are often redacted or encrypted, and
  storing an agent's raw reasoning adds privacy exposure without helping
  appraisal, which uses outcomes and visible messages.
- Kernle-originated content is dropped first, so Kernle never treats its own
  recall as new experience.
