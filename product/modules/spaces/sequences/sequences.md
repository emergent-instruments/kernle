# Spaces sequences

The spaces module owns the Kernle home (`ARCH-009`), the space registry
([DATA-001](../data/data-model.md)), owner-only access, runtime connection, and
the daemon lifecycle (`ARCH-008`). Rust module `spaces` in the `kernle` binary
owns these processes.

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [ResolveSpace](#seq-003-resolve-space-and-authorize-a-request) | Every hook, MCP, plugin, and CLI request | Refusal unless the caller is the home's owner. Otherwise the request's primary space and readable spaces. Agent requests can never read spaces outside the current project and the personal space |
| Runtime connection record | [capture SEQ-010](../../capture/sequences/sequences.md#seq-010-handle-a-runtime-hook-event), [consolidation SEQ-031](../../consolidation/sequences/sequences.md#seq-031-sleep-run) | `config.toml` `runtimes.<runtime>` keys from [config contract](../contracts/config.yaml). `dreamer_allowed = false` means no episode from that runtime is sent to the Dreamer |
| Daemon availability | All modules | Exactly one daemon per Kernle home. Space databases are written only by the daemon while it runs |

## SEQ-001 Initialize Kernle home

Runs from [FLOW-001](../experience/user-flows.md#flow-001-set-up-kernle-and-connect-a-runtime).
`kernle init` is idempotent: it creates only what is missing.

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
  participant H as ARCH-009 Kernle home
  participant SM as OS service manager
  participant M as ARCH-012 Model source
  participant K as OS keychain

  O->>C: kernle init
  Note over C: OWNER · spaces::init()
  C->>H: Inspect ~/.kernle
  alt Home exists and is owned by another OS account
    H-->>C: owner uid differs from caller uid
    C-->>O: Refuse · nothing created or changed
  else Home missing or owned by caller
    C->>H: Create missing dirs (mode 0700) and config.toml (mode 0600)
    Note right of H: WRITE · ~/.kernle, spaces/, run/, models/<br/>config.toml defaults from config contract<br/>PRESERVE · existing values are never overwritten
    C->>H: Ensure the personal space exists
    Note right of H: INSERT if absent · DATA-001 space in registry.db<br/>CONSTRAINT · [U1] space_root_key_key<br/>INPUT · kind = 'user', root_key = 'user'<br/>WRITE · spaces/{id}/ with an empty kernle.db at the binary's schema
    C->>SM: Install and load the kernled user service
    alt Service installed
      SM-->>C: loaded
    else Install or load fails
      SM-->>C: error
      C-->>O: Show manual start command · setup continues
    end
    C->>M: Download the pinned embedding model
    Note right of M: INPUT · URL and SHA-256 from constants in the binary<br/>NO WRITE · nothing about the owner is sent
    alt Checksum matches
      C->>H: Store models/{model_id} and set embedding.available = true
    else Download fails or checksum differs
      C->>H: Discard the file and set embedding.available = false
      C-->>O: Show keyword-only recall until kernle init runs again
    end
    O->>C: Choose Dreamer destination (FLOW-001)
    alt Provider uses an API key
      C->>K: Store the key under service ai.kernle.dreamer
      Note right of K: Linux without a keychain: ~/.kernle/credentials mode 0600<br/>NO WRITE · keys never go into config.toml
    end
    C->>H: Write dreamer.provider, dreamer.model, dreamer.endpoint
    C->>SM: Start kernled
    SM-->>C: started, continues with SEQ-004
    C-->>O: Show setup complete
  end
```

### Current rationale

- Init is idempotent and never overwrites existing values. Owners rerun it to
  recover from a failed download or service install, and rerunning must not
  reset choices.
- The model checksum is pinned in the binary. Otherwise a compromised download
  could substitute a model that runs inside the daemon on every recall.
- API keys live in the OS keychain because `config.toml` is meant to be readable
  and editable. Keys in it would leak through backups, screenshots, and agents
  that read files.

## SEQ-002 Connect a runtime

Runs from [FLOW-001](../experience/user-flows.md#flow-001-set-up-kernle-and-connect-a-runtime).

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
  participant R as Runtime install (ARCH-001, 002, or 003)
  participant H as ARCH-009 Kernle home
  participant D as ARCH-008 kernled

  O->>C: kernle connect claude-code | codex | openclaw
  Note over C: OWNER · spaces::connect()
  C->>R: Detect the runtime binary and its config directory
  alt Runtime not found
    C-->>O: Show where Kernle looked · nothing installed
  else Runtime found
    C->>H: Read dreamer.provider
    Note right of H: GUARD · cross-provider when the Dreamer is remote<br/>and its provider family differs from the runtime's:<br/>claude-code is anthropic, codex is openai,<br/>openclaw is other<br/>Local endpoints and None are never cross-provider
    alt Cross-provider
      C->>O: Ask to allow sending this runtime's sessions to the Dreamer
      O-->>C: Allow or Decline
      Note over C: INPUT · dreamer_allowed from the owner's answer
    else Not cross-provider
      Note over C: INPUT · dreamer_allowed = true
    end
    C->>R: Install the adapter
    Note right of R: claude-code · Kernle plugin ARCH-005 at user scope<br/>(hooks and MCP server kernle mcp)<br/>codex · MCP server entry kernle mcp in Codex config<br/>openclaw · Kernle plugin ARCH-006, enabled
    alt Install fails
      R-->>C: error
      C->>R: Remove any partially installed adapter files
      C-->>O: Show error · runtime not connected
    else Installed
      C->>H: Write runtimes.{runtime} = connected_at, dreamer_allowed
      C-->>O: Show installed adapter and restart hint
      O->>C: Choose Import or Skip
      opt Import
        C->>D: capture ImportHistory(runtime, since = now - 30 days)
        Note over D: Runs capture SEQ-012 and streams progress to the CLI
        D-->>C: sessions read, episodes found, next sleep time
      end
      C-->>O: Show connected
    end
  end
```

### Current rationale

- Kernle installs a declared adapter (plugin or MCP entry), not hand-edited
  hook settings, because runtimes manage and update their own plugin and MCP
  configuration. Owners can then also remove Kernle with the runtime's tools.
- A failed install removes partial adapter files. A half-installed hook
  configuration can call Kernle in a broken way from inside the agent's
  critical path.
- The cross-provider choice is recorded per runtime because the privacy
  question is "who may see this runtime's sessions", and runtimes differ.

## SEQ-003 Resolve space and authorize a request

Every request that reads or writes memory first passes through this process:
hook handling, MCP tools, the OpenClaw plugin, and owner commands. Rust
function `spaces::resolve()` owns it.

**DCL override:** 4 — this step runs inside the agent's hook deadline, so it must
answer quickly and fail open on every call.

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
  participant Q as Requester (ARCH-006 or ARCH-007)
  participant D as ARCH-008 kernled
  participant G as Git (local repository)
  participant DB as ARCH-009 registry.db

  Q->>D: Request with role, runtime, cwd, session_id
  Note right of D: INPUT · peer uid from the Unix socket credentials<br/>role from the entry point: owner_cli, agent_hook,<br/>agent_mcp, or openclaw_plugin<br/>cwd and session_id from the runtime's hook or MCP payload
  alt Peer uid differs from the owner uid of ~/.kernle
    D-->>Q: Refuse · no memory, metadata, or counts returned
  else Owner's account
    D->>DB: Ensure the personal space exists
    Note right of DB: READ · DATA-001 space · ACCESS · [U1] with root_key 'user'<br/>INSERT when missing (deleted by retention SEQ-062 after a year idle)
    alt runtime = openclaw, or cwd is not inside a git repository
      Note over D: RETURN · primary = personal space<br/>readable = personal space
    else cwd inside a git repository
      D->>G: Find the toplevel and the origin remote URL
      Note right of G: root_key = 'git:' + origin URL normalized<br/>(scheme, credentials, trailing .git removed, host lowercased)<br/>or 'path:' + toplevel when there is no origin<br/>Worktrees of one repository share a root_key
      D->>DB: Find the project space
      Note right of DB: READ · DATA-001 space<br/>ACCESS · [U1] space_root_key_key<br/>INPUT · root_key from the step above
      alt Space exists
        DB-->>D: space
      else No space yet
        D->>DB: Create the project space and its directory
        Note right of DB: INSERT · DATA-001 space<br/>WRITE · id new ULID, kind 'project',<br/>display_name from the repository folder name<br/>CONSTRAINT · [U1] a concurrent create resolves to one row
      end
      Note over D: RETURN · primary = project space<br/>readable = project space and personal space
    end
    opt Request is capture or recall activity, and last_active_at is older than 1 hour
      D->>DB: Mark the returned spaces active
      Note right of DB: UPDATE · DATA-001 space.last_active_at = now<br/>KEY · id of each returned space
    end
    Note over D: GUARD · role owner_cli may name any space explicitly with --space<br/>Agent roles receive only the readable set and cannot name another space
    D-->>Q: primary space and readable spaces
  end
```

### Current rationale

- The access boundary is the OS account that owns `~/.kernle`. Every Kernle
  file is readable only by that account, and the daemon rejects other socket
  peers. Agents run under the owner's account, so an agent with shell access
  can reach anything the owner can. Kernle's owner and agent roles therefore
  set default behavior, not a security wall. The guards that matter against a
  misbehaving agent live in the processes that write or delete memory.
- An agent sees only the current project's space and the personal space.
  Otherwise memories from one client's repository could appear in another
  client's work.
- The project identity comes from the origin remote. Separate clones and
  worktrees of the same repository should share their experience, and a
  folder rename should not lose it.
- OpenClaw sessions use the personal space because a personal agent's
  experience concerns the person, not a repository.
- `last_active_at` is updated at most hourly, because idle deletion needs only
  day precision and a write on every hook call would contend with the daemon's
  other work.

## SEQ-004 Daemon start and catch-up

Runs whenever the service manager starts `kernled`: at login or boot, after
`kernle init`, or after `kernle daemon start`. Rust function `daemon::run()`
owns it.

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
  participant SM as OS service manager
  participant D as ARCH-008 kernled
  participant H as ARCH-009 Kernle home

  SM->>D: Start
  D->>H: Take an exclusive lock on run/kernled.lock
  alt Lock held by another kernled
    D-->>SM: Exit · the running daemon continues
  else Lock acquired
    D->>H: Open registry.db and each space's kernle.db
    Note right of H: Each database is brought to the schema declared by this binary<br/>GUARD · a space whose database cannot be opened or upgraded<br/>is marked unavailable for this run and shown in kernle status<br/>PRESERVE · its files are left untouched
    D->>D: memory RegisterOwnerEdits in reconcile mode (memory SEQ-041)
    D->>D: recall MaintainIndex, rebuilding when missing or the embedder changed (recall SEQ-055)
    D->>D: capture IngestSource for each connected runtime source changed since its cursor (capture SEQ-011)
    D->>D: consolidation start the maintenance scheduler (consolidation SEQ-030)
    D->>H: Bind run/kernled.sock (mode 0600)
    Note over D: Serves requests only after catch-up has been queued<br/>Requests during catch-up see the memory as of the last run
  end
```

### Current rationale

- One daemon per home, enforced with a file lock, because two daemons would be
  two writers to every space database. Duplicated runs would create duplicate
  memories.
- Catch-up ingestion at start lets a missed hook or a stopped daemon delay
  memory without losing it. The runtimes keep their session records, so
  nothing needs to be replayed from Kernle's side.
- A space that cannot be upgraded stays untouched and visible as unavailable.
  Repairing it automatically could destroy memory, so the owner must see it
  instead.
