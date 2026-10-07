# Memory user flows

## FLOW-004 Read, edit, and pin memory as files

The owner works with memory in any editor or file browser. The watcher
registers every change through
[SEQ-041](../sequences/sequences.md#seq-041-register-owner-edits).

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
  START(["ACTOR-001 opens ~/.kernle/spaces/&lt;space&gt;/cards or skills"])
  subgraph S_FILES["EXTERNAL · owner's editor or file browser"]
    F1["Browse cards by kind folder: preference, lesson, guardrail, fact, open-loop, skills"]
    F2["Read a card: rule, Why line with dated evidence, Applies when line, appraisal, and sources"]
    F3["Edit body, name, description, or triggers, or set kernle.pinned: true, then save"]
    F4["Delete a card file"]
    F5["Create a new card file"]
  end
  subgraph S_RESULT["SURFACE · kernle log and kernle status"]
    R1["kernle log shows 'owner edited', 'pinned', 'deleted by owner', or 'added by owner'"]
    R2["kernle status names a card file whose frontmatter does not parse<br/>and says the last valid version is still in use"]
  end
  START --> F1 --> F2
  F2 --> F3
  F2 --> F4
  F1 --> F5
  F3 -->|valid frontmatter| R1
  F3 -->|frontmatter invalid| R2
  R2 -->|owner fixes and saves| R1
  F4 --> R1
  F5 --> R1
```

## Current rationale

- Memory is plain files so the owner never needs a special tool to see or
  change what their agents remember. Trust in a memory system comes from being
  able to open it.
- Every edit shows up in the log. The owner can then confirm that Kernle saw
  the change, and later sleeps treat that card as the owner's own words.
