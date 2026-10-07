# Runtime sequences

[Stack context](../architecture/stack-context.md) owns placement and the trust
boundary. [The data model](../data/data-model.md) owns origin, ownership,
retention, and field constraints. [User flows](../experience/user-flows.md)
own visible outcomes. Each sequence uses the same Entity/Stack policy path.

## Common operation contract

The trusted host supplies authenticated requester, owner, stack binding, and
operation context. The public request supplies only the operation's content,
selectors, and logical request identity. Caller text is never authentication.
The runtime refuses omitted or forged identity before returning any protected
content or metadata. The owner-only decision applies to all retained surfaces,
including Stack calls, diagnostic output, serialization, and audit inspection.

Stable machine outcomes distinguish invalid input, unauthenticated requester,
access denial, missing record, empty results, identity/request conflict,
unavailable storage, invalid artifact, unsupported format/schema, and successful
commit or replay. Human output gives an applicable correction or retry action.
These are semantic outcomes; transport formatting belongs to the host adapter.

For mutations, the logical request identity comes from the host/API request and
is correlated through `DATA-AUDIT / memory_audit.correlation_id`. Audit `details`
retains the operation fingerprint and resulting identities without duplicating
raw content. Replay checks
match the operation, selected stack, and original content/effects. An identical
committed request returns its existing result; conflicting reuse fails. The
check and mutation share the selected stack's short write transaction so racing
requests cannot each create the same logical result. Arbitrary repeated content
with a new request identity is not silently discarded as duplicate experience.

Every denied memory operation produces a minimal owner-visible security audit
with the attempted operation, reason, correlation identity, and authenticated
actor or unauthenticated marker. It contains no copied request content. When a
trusted stack binding is known, the runtime appends that event to `DATA-AUDIT /
memory_audit` independently of the refused memory mutation. If no bound store
is available or audit persistence fails, the trusted host receives a safe
diagnostic on its local operator output; access remains denied. Successful
reads need no blanket access log. Each refusal branch below applies this rule.

No automatic unbounded retry is required. When an outcome is uncertain, retry
with the original request identity so the runtime can reconcile a prior commit.
A caller corrects invalid input or ownership before retrying. A denied operation
never becomes allowed because audit logging or storage is unavailable.

## SEQ-CAPTURE — durable raw evidence

**DCL:** 3 (product default).

Owners: [`Entity.raw`](../../../../kernle/entity.py),
[`Stack.save_raw`](../../../../kernle/stack/sqlite_stack.py), and
[`raw_entries.save_raw`](../../../../kernle/storage/raw_entries.py).

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
---
sequenceDiagram
  autonumber
  participant H as Trusted host / Python caller
  participant R as Kernle runtime
  participant D as Selected SQLite database
  H->>R: Capture raw input with authenticated context and request identity
  Note right of R: INPUT · blob from caller<br/>owner and stack from trusted binding<br/>source from ingress adapter, never identity
  R->>R: Authorize owner and validate complete input
  alt Invalid or unauthorized
    R-->>H: Stable failure, no protected content, no memory write
  else Authorized valid capture
    R->>D: Begin write transaction and check committed request
    Note right of D: READ · DATA-AUDIT / memory_audit<br/>KEY · correlation_id plus operation<br/>scope is the selected stack database
    alt Identical committed request
      D-->>R: Existing raw identity and recorded result
      R->>D: End transaction without mutation
      R-->>H: Idempotent replay
    else Request identity conflicts
      R->>D: End transaction without mutation
      R-->>H: Request conflict
    else New request
      R->>D: Insert complete raw record and capture audit
      Note right of D: INSERT · DATA-RAW / raw_entries<br/>id, stack_id, blob, captured_at, source<br/>INSERT · DATA-AUDIT / memory_audit<br/>actor from authenticated context, correlation_id from request
      alt All required changes commit
        D-->>R: Durable raw identity
        R-->>H: Capture success
      else Write or commit fails
        R->>D: Roll back operation
        R-->>H: Storage failure, caller retains input for retry
      end
    end
  end
```

Validation refuses unsupported input rather than silently truncating evidence.
The durable alpha record lives in SQLite; an auxiliary file or index cannot be
an independently acknowledged capture. A derived index failure cannot produce
a successful but unretrievable logical record. Input content is never executed
as a policy instruction.

## SEQ-PROMOTE — explicit raw-to-episode interpretation

**DCL:** 3 (product default).

[`Entity`](../../../../kernle/entity.py) owns the explicit promotion operation.
[`Stack`](../../../../kernle/stack/sqlite_stack.py) owns authorization and provenance
validation; [`episodes_crud`](../../../../kernle/storage/episodes_crud.py) owns episode
persistence within the operation's transaction. Promotion has no model call or
automatic cognitive pipeline.

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
---
sequenceDiagram
  autonumber
  participant H as Trusted host / Python caller
  participant R as Kernle runtime
  participant D as Selected SQLite database
  H->>R: Explicit promotion with source IDs and episode content
  Note right of R: INPUT · source IDs, objective, outcome and optional lessons from<br/>caller<br/>origin identity from authenticated context<br/>request identity retained across retries
  R->>R: Authorize owner and validate retained type and fields
  alt Invalid or unauthorized
    R-->>H: Refuse without changing source evidence
  else Valid explicit request
    R->>D: Begin transaction and reconcile request identity
    Note right of D: READ · DATA-AUDIT / memory_audit by correlation_id and operation<br/>Identical replay returns prior result, conflicting reuse refuses
    break Request is already committed or its identity conflicts
      R->>D: End transaction without mutation
      R-->>H: Return existing result or explicit conflict
    end
    R->>D: Read selected saved source records
    Note right of D: READ · DATA-RAW / raw_entries<br/>KEY · each id and selected stack_id<br/>GUARD · every source exists and is accessible
    alt Missing source, foreign stack, or invalid provenance
      R->>D: Roll back without memory changes
      R-->>H: Source or provenance failure
    else All source and target constraints hold
      R->>D: Insert episode, append raw derivation markers, and append audit
      Note right of D: INSERT · DATA-EPISODE / episodes<br/>objective, outcome and lessons from explicit request<br/>derived_from from validated raw IDs, source_entity from trusted<br/>context<br/>UPDATE · DATA-RAW / raw_entries.processed and processed_into by<br/>source IDs<br/>INSERT · DATA-AUDIT / memory_audit with original request identity
      alt All changes commit
        D-->>R: Episode identity and preserved raw references
        R-->>H: Promotion success
      else Any write or commit fails
        R->>D: Roll back source, target and audit changes together
        R-->>H: Promotion failure, saved raw evidence remains intact
      end
    end
  end
```

The caller's explicit invocation is the promotion decision. No second human
approval queue or background promotion is implied. A direct episode save that
omits the same authorization and saved-evidence checks is not another public
write path. Unsuccessful promotion never marks its source as successfully
processed.

## SEQ-RETRIEVE — authorized get and text search

**DCL:** 3 (product default).

[`Entity.search`](../../../../kernle/entity.py) and
[`Stack.search`](../../../../kernle/stack/sqlite_stack.py) own routed search; the Stack
owns direct getters through the same policy boundary.

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
---
sequenceDiagram
  autonumber
  participant H as Trusted host / Python caller
  participant R as Kernle runtime
  participant D as Selected SQLite database
  H->>R: Get record identity or search text and result limit
  R->>R: Validate selectors and authorize authenticated owner
  alt Invalid or unauthorized
    R-->>H: Safe stable reason, no records, snippets or counts
  else Authorized query
    R->>D: Read retained records within selected stack
    Note right of D: READ · DATA-RAW / raw_entries by id + stack_id or blob text<br/>READ · DATA-EPISODE / episodes by id + stack_id or<br/>objective/outcome/lessons text<br/>limit from validated caller input, stable ordering includes<br/>record identity
    alt Storage unavailable
      R-->>H: Retrieval failure with retry guidance
    else Query completes
      D-->>R: Matching records and source references
      R->>D: Read origin events for returned raw records and requested audit details
      Note right of D: READ · DATA-AUDIT / memory_audit
      Note right of D: KEY · memory_type and memory_id within selected stack database
      D-->>R: Authorized capture actors and provenance metadata
      R->>R: Apply owner-only policy to complete returned projection
      alt No matches
        R-->>H: Empty search or record-not-found result
      else Authorized matches exist
        R-->>H: Typed records with origin and evidence references
      end
    end
  end
```

Text retrieval is deterministic for the same saved state and query. It does not
require embeddings, network fallback, or a semantic-quality claim. Source links
and audit details cannot expose data that the caller could not retrieve
directly. Search cannot change the selected stack based on memory content.

## SEQ-PORTABILITY — one export and restore workflow

**DCL:** 3 (product default).

[`Stack.export`](../../../../kernle/stack/sqlite_stack.py) owns export orchestration;
[`Entity`](../../../../kernle/entity.py) owns restore orchestration through Stack
policy. [`serializers`](../../../../kernle/core/serializers.py) and
[`importers`](../../../../kernle/importers/) supply serialization and parsing within
that boundary. Normal restore has no origin-repair capability.

```mermaid
---
config:
  theme: dark
  sequence:
    noteAlign: left
---
sequenceDiagram
  autonumber
  participant O as Authenticated owner
  participant R as Kernle runtime
  participant D as Selected SQLite database
  participant F as Local filesystem
  O->>R: Export selected stack to local destination
  R->>R: Authorize owner and validate destination
  break Unauthorized request or invalid destination
    R-->>O: Refuse without exposing or changing memory
  end
  R->>D: Read one consistent authorized memory snapshot
  Note right of D: READ · DATA-RAW / raw_entries and DATA-EPISODE / episodes by<br/>stack_id<br/>READ · DATA-AUDIT / memory_audit for retained origin and lineage<br/>events<br/>No protected content or metadata bypasses policy
  break Snapshot read or projection authorization fails
    R-->>O: Safe export failure, no artifact published
  end
  D-->>R: Records, references, origin and stack binding
  R->>F: Write complete artifact without overwriting unrelated files
  alt Export fails
    R-->>O: Export failure, no complete-artifact claim
  else Export completes
    R-->>O: Completed export location
  end
  O->>R: Preview artifact, destination and logical restore request identity
  R->>R: Authenticate owner and authorize restore preview
  break Missing identity or access denied
    R-->>O: Safe refusal, destination unchanged
  end
  R->>F: Read artifact as untrusted input
  F-->>R: Bytes and format identification
  R->>R: Validate format, ownership, all records and lineage
  R->>D: Reconcile restore request against committed audit in destination
  Note right of D: READ · DATA-AUDIT / memory_audit by correlation_id and operation
  break Identical committed restore or conflicting request identity
    R-->>O: Return prior restore result or explicit conflict without mutation
  end
  R->>D: Check destination identity and emptiness
  Note right of D: Destination binding must match trusted owner and stack
  Note right of D: READ · raw_entries and episodes for existing records
  alt Invalid, unsupported, foreign owner or populated destination
    R-->>O: Refuse, destination unchanged
  else Valid preview
    R-->>O: Destination, included records and planned effects
    O->>R: Explicitly commit preview with the same restore request identity
    R->>R: Reauthorize owner and reconcile original restore request
    break Unauthorized, committed replay, or conflicting request identity
      R-->>O: Safe denial, existing result, or explicit conflict
    end
    R->>R: Recheck artifact and destination against preview
    alt Preview no longer matches
      R-->>O: Require a fresh preview, no mutation
    else Preview remains valid
      R->>D: Begin transaction and validate destination again
      R->>D: Restore records and original audit evidence, append restore audit
      Note right of D: INSERT · DATA-RAW / raw_entries<br/>INSERT · DATA-EPISODE / episodes<br/>INSERT · DATA-AUDIT / memory_audit<br/>Preserve IDs, times, origin and lineage, current actor owns<br/>restore event only
      alt Complete restore commits
        D-->>R: Durable restore result
        R-->>O: Restore success
      else Constraint, write or commit failure
        R->>D: Roll back complete restore
        R-->>O: Failure, previous destination state preserved
      end
    end
  end
```

The commit remains owner-authorized even after preview. Repeating the same
completed restore request returns its recorded result; it does not merge the
artifact into an already populated destination. Missing lineage, duplicate or
conflicting identities, unsupported retained types, and incompatible owner
claims are invalid artifacts. A complete preview is not permission to repair
origin, transfer ownership, erase records, or overwrite existing memory.

## Current rationale

Authentication precedes both reads and writes because filtering only a primary
query leaves alternate surfaces and metadata exposed. One transaction binds
promotion to its source and audit evidence because partial writes could
otherwise claim an interpretation exists when its evidence or origin is absent.
Request reconciliation is necessary because a host can lose a successful reply
and retry. Explicit promotion makes the local workflow useful without depending
on an inference service. Snapshot export and validated atomic restore preserve
a coherent evidence graph across local installations.
