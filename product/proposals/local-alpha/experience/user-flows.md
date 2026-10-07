# User flows

The product surfaces are Python calls, MCP tools, and a small operator CLI.
There is no visual application or binding visual mockup. Each bounded surface
below defines its observable states. [Runtime sequences](../sequences/sequences.md)
own authorization, persistence, and recovery mechanics.

## FLOW-SETUP — establish a usable stack

Supports [CAP-SETUP](../product.yaml).

```mermaid
---
config:
  theme: dark
---
flowchart TD
  subgraph cli["SURFACE · Operator CLI"]
    start["Choose stack identity and local<br/>destination"]
    init["Initialize or reopen selected stack"]
    ready["See selected stack and empty or<br/>populated state"]
    error["See identity, path, or storage error<br/>and next action"]
    inspect["Inspect status or diagnostics"]
    start --> init
    init -->|"Usable"| ready
    init -->|"Cannot open safely"| error
    error -->|"Correct configuration and retry"| init
    ready --> inspect
  end
  subgraph host["SURFACE · Local MCP host or trusted<br/>Python application"]
    bind["Attach the owner-bound stack"]
    usable["Use available memory operations"]
    denied["See unavailable or unauthenticated<br/>reason"]
    bind -->|"Attached"| usable
    bind -->|"Cannot attach"| denied
  end
  ready --> bind
```

## FLOW-MEMORY — capture, interpret, and retrieve

Supports [CAP-CAPTURE and CAP-RETRIEVE](../product.yaml).

```mermaid
---
config:
  theme: dark
---
flowchart TD
  subgraph surface["SURFACE · MCP tools or canonical<br/>Python API"]
    capture["Submit raw evidence"]
    saved["Receive durable raw identity"]
    captureError["See capture failure; submitted<br/>evidence remains available to retry"]
    select["Select saved raw evidence and enter<br/>episode objective and outcome"]
    promote["Explicitly request episode promotion"]
    episode["Receive episode identity and source<br/>references"]
    promotionError["See promotion refusal or failure;<br/>saved evidence remains intact"]
    lookup["Get a record or search by text"]
    results["Inspect matching evidence, episodes,<br/>and provenance"]
    empty["See empty result or record not found"]
    readError["See invalid, unauthorized, or<br/>unavailable result"]
    capture -->|"Saved"| saved
    capture -->|"Failed"| captureError
    captureError -->|"Retry unchanged input"| capture
    saved --> select --> promote
    promote -->|"Committed"| episode
    promote -->|"Not committed"| promotionError
    promotionError -->|"Correct input or retry"| select
    saved --> lookup
    episode --> lookup
    lookup -->|"Matches"| results
    lookup -->|"No matching memory"| empty
    lookup -->|"Cannot retrieve"| readError
    readError -->|"Correct context or retry"| lookup
  end
```

A denial never includes another owner's memory. A repeated committed request
returns its original result; it is distinguishable from a new write and from a
conflicting request. Structured errors identify whether the caller must correct
input, fix ownership or configuration, or retry storage access. No processing
or model-configuration step is required.

## FLOW-PORTABILITY — inspect an export and restore it

Supports [CAP-PORTABILITY](../product.yaml).

```mermaid
---
config:
  theme: dark
---
flowchart TD
  subgraph surface["SURFACE · Canonical export and restore<br/>API"]
    request["Choose local export destination"]
    exported["Receive completed export location"]
    inspect["Inspect owner-authorized records and<br/>provenance"]
    preview["Select artifact and empty destination;<br/>request restore preview"]
    effects["See destination identity, included<br/>records, and validation result"]
    choose{"Commit this preview?"}
    restored["Receive durable restore result"]
    canceled["Leave destination unchanged"]
    failure["See safe reason; prior destination<br/>remains intact"]
    reconcile["Retry uncertain outcome with the<br/>original request identity"]
    request -->|"Exported"| exported --> inspect --> preview
    request -->|"Export failed"| failure
    preview -->|"Valid"| effects --> choose
    preview -->|"Invalid, unauthorized, or conflicting"| failure
    choose -->|"No"| canceled
    choose -->|"Yes"| restored
    choose -->|"Commit fails or preview is outdated"| failure
    failure -->|"Correct known input or destination failure"| preview
    failure -->|"Commit outcome uncertain"| reconcile
    reconcile -->|"Already committed"| restored
    reconcile -->|"No commit; valid preview"| choose
    reconcile -->|"Conflict or changed preview"| preview
  end
```

## Current rationale

The CLI concentrates on setup and diagnosis because a second application
command surface would duplicate memory behavior. MCP and Python expose the same
memory outcomes. Capture and explicit promotion are separate choices because
saving evidence must not silently turn it into an interpretation. Restore has a
preview and explicit commit because the owner needs to understand the local
state change before authorizing it. Empty, denied, and failed outcomes stay
distinct so the caller can choose a useful next action.
