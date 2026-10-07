# Memory sequences

Memory owns cards and skills: the markdown files
([DATA-011](../data/data-model.md)), their registry, versions, provenance, and
change ledger ([DATA-007 to DATA-010](../data/data-model.md)), skill projection
into runtime skill directories (`ARCH-010`), and the changelog projection. The
file format is in [card-format.yaml](../contracts/card-format.yaml). Rust module
`cards` owns these processes.

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [ApplyCardOp](#seq-040-apply-a-card-operation) | consolidation SEQ-031 and SEQ-032, capture SEQ-013, retention SEQ-060 and SEQ-061 | Returns `applied` (with the new version), `conflict` (expected version differs, nothing written), or `refused` (with a reason, nothing written). Idempotent per operation id. The registry, version, provenance, and ledger commit together, and the file always converges to the committed version |
| [RegisterOwnerEdits](#seq-041-register-owner-edits) | file watcher, daemon start (spaces SEQ-004), `kernle review` Edit | Owner changes to card files become owner-authored versions. Owner-deleted files become deleted cards. Kernle-managed fields keep Kernle's values |
| [WriteChangelog](#seq-042-write-the-changelog) | consolidation SEQ-031, retention SEQ-061 | Regenerates the changelog files for affected dates from the ledger. Deleted cards appear without their names |
| Card read projection | recall, retention, consolidation | `DATA-007 card` rows plus the current file content. Usage columns are written by recall and `strength` by retention ([field ownership](../data/data-model.md#field-ownership-on-card)) |

## SEQ-040 Apply a card operation

Rust function `cards::apply_op()` owns every Kernle-initiated card change.

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
  participant Q as Caller (sleep run, review, remember, or retention)
  participant M as ARCH-008 cards::apply_op
  participant DB as ARCH-009 space kernle.db
  participant FS as ARCH-009 card files
  participant SK as ARCH-010 ~/.claude/skills

  Q->>M: ApplyCardOp(op_id, op, kind, target_card_id, expected_version, proposal, sources, author)
  Note right of M: INPUT · op_id from DATA-006 card_op.id, or a new ULID for owner and retention calls<br/>author 'sleep' or 'owner' from the caller<br/>sources: cited event and episode ids, each with its space id
  M->>DB: Check whether this operation already applied
  Note right of DB: READ · DATA-008 card_version<br/>ACCESS · DATA-008 card_version [P1] card_version_op_key<br/>KEY · op_id
  alt Already applied
    M-->>Q: applied (the recorded version) · no write
  else New operation
    opt op is not add
      M->>DB: Load the target card
      Note right of DB: READ · DATA-007 card · KEY · id = target_card_id
    end
    alt Target missing (deleted since the proposal)
      M-->>Q: refused · target missing · no write
    else Target current_version differs from expected_version
      M-->>Q: conflict · no write
    else Proposal invalid (body empty or over 2000 characters, name empty)
      M-->>Q: refused · reason · no write
    else Valid
      M->>M: Compose the resulting card files
      Note over M: add · new id, name = unique slug of proposal name, version 1<br/>update · new body, description, triggers, confidence, version + 1<br/>merge · target absorbs the merged card, which becomes superseded with superseded_by<br/>supersede · a new card is added, and the target becomes superseded with valid_to = now<br/>reinforce · no file change, last_used_at = now, confidence + 0.05 capped at 1<br/>retire · target becomes superseded with valid_to = now and no replacement (closed open loops)<br/>set_status · active or dormant, decided by retention RULE-040, version + 1<br/>delete · no new content (retention SEQ-060 and SEQ-061, review of a held agent forget)
      M->>DB: Commit the change in one transaction
      Note right of DB: INSERT · DATA-008 card_version (full file content, content_hash, author, op_id)<br/>UPSERT · DATA-007 card (current_version, file_path, kind, trust = highest source trust,<br/>salience, valence, arousal, label from the cited episodes, confidence, triggers, status)<br/>INSERT · DATA-009 card_source (relation evidence, merged_from, or supersedes)<br/>INSERT · DATA-010 card_ledger (change, version, reason, card_name, run_id)<br/>delete · DELETE DATA-008, DATA-009, DATA-007 rows for the card<br/>UPDATE · DATA-010 card_ledger card_name = NULL for every row of the card<br/>TRANSACTION · all rows for this operation commit together<br/>CONSTRAINT · DATA-008 card_version [P1] and DATA-007 card [U1] card_name_key
      alt Commit fails
        M-->>Q: refused · storage error · nothing written
      else Committed
        M->>FS: Write or remove the card files
        Note right of FS: Write to a temporary file in the same directory (mode 0600), fsync, rename<br/>cards/{kind}/{name}.md, or skills/{name}/SKILL.md for skills<br/>delete · unlink the file or skill folder<br/>A crash before this step is repaired by SEQ-041 reconcile
        opt kind skill, scope user, and Claude Code connected
          M->>SK: Project, update, or remove ~/.claude/skills/kernle-{name}/
          Note right of SK: The folder holds SKILL.md and a .kernle-generated marker<br/>GUARD · only folders carrying the marker are ever changed or removed<br/>WRITE · DATA-007 card.projected_path
        end
        M-->>Q: applied · new version
      end
    end
  end
```

### Current rationale

- The database commits before the file is written. The database version is
  authoritative for Kernle's own writes, and SEQ-041 can then tell a crash
  apart from an owner edit by comparing the file with the last two versions.
- Expected versions make changes optimistic (`COORD-002`). Sleep, review,
  remember, retention, and owner edits can all change a card, and a check at
  write time protects each card without serializing unrelated work.
- Only project-independent skills (scope user) are projected into Claude
  Code. Writing project skills into a repository's `.claude/` folder would
  change the owner's repository. Project skills reach the agent through recall
  instead.
- Projection only touches folders carrying Kernle's marker. A name collision
  then cannot overwrite a skill the owner wrote by hand.

## SEQ-041 Register owner edits

Triggered by the card file watcher (`notify`, events debounced for 2 seconds),
by reconcile mode at daemon start (spaces SEQ-004), and by `kernle review`
Edit. Rust function `cards::register_edits()` owns it.

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
  participant W as ARCH-008 watcher or reconcile
  participant FS as ARCH-009 card files
  participant DB as ARCH-009 space kernle.db

  W->>FS: Read the changed path, or every card file in reconcile mode
  W->>DB: Find the card by frontmatter kernle.id, else by file_path
  Note right of DB: READ · DATA-007 card and its two latest DATA-008 card_version rows
  alt File removed by the owner
    W->>W: memory ApplyCardOp(delete, author owner) for the card (SEQ-040)
    Note over W: The owner's deletion is a forget without cascade<br/>Episodes and other cards are left unchanged
  else File hash equals card_version current content_hash
    Note over W: In sync · no write
  else File hash equals the previous version's hash
    W->>FS: Rewrite the file from the current version
    Note over W: A Kernle write was interrupted after commit (SEQ-040)
  else New file without a kernle.id
    W->>W: memory ApplyCardOp(add, author owner, no sources) (SEQ-040)
    Note over W: The resulting file gains kernle.id and Kernle-managed fields<br/>trust principal
  else Frontmatter does not parse, or required fields are missing
    W->>DB: Ledger entry 'owner_edit_invalid'
    Note right of DB: INSERT · DATA-010 card_ledger<br/>PRESERVE · the card keeps its last valid version for recall<br/>kernle status names the file until it parses
  else Valid owner edit
    W->>W: memory ApplyCardOp(update, author owner, expected_version = current) (SEQ-040)
    Note over W: Owner-editable: name, description, body, pinned, triggers<br/>Kernle-managed fields (id, kind, scope, sources, appraisal, trust)<br/>keep their registry values in the new version<br/>The new version's trust is principal<br/>pinned changes add 'pinned' or 'unpinned' ledger entries
  end
```

### Current rationale

- Files are the owner's interface. Editing or deleting a card in any editor
  must work without a Kernle command, and the watcher makes that the normal
  path.
- Deleting a file forgets only that card. The owner did not see a cascade
  preview, so Kernle must not remove anything else on their behalf.
- Invalid frontmatter keeps the last valid version in use. A half-saved file
  or a typo must not make a memory vanish from recall.
- Kernle-managed fields are restored after an owner edit. Provenance and
  appraisal must stay truthful, and the owner changes meaning through the
  body, name, triggers, and pin.

## SEQ-042 Write the changelog

Called at the end of each sleep run (consolidation SEQ-031) and after
deletions (retention SEQ-061). Rust function `cards::write_changelog()` owns it.

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
  participant Q as Caller (sleep run or retention)
  participant M as ARCH-008 cards::write_changelog
  participant DB as ARCH-009 space kernle.db
  participant FS as ARCH-009 changelog files

  Q->>M: WriteChangelog(run_id or affected dates)
  M->>DB: Read ledger entries and held or refused operations for the affected dates
  Note right of DB: READ · DATA-010 card_ledger [I1] card_ledger_created_idx<br/>READ · DATA-006 card_op rows for the same runs<br/>READ · DATA-005 maintenance_run counters
  M->>FS: Rewrite changelog/{YYYY-MM-DD}.md for each affected date (mode 0600)
  Note right of FS: Per run: learned, updated, forgot, held, and refused counts,<br/>then one line per change: card name, kind, change, and reason<br/>A NULL card_name renders as 'a {kind} card (deleted)'<br/>PRESERVE · dates not affected are left unchanged
```

### Current rationale

- The changelog is regenerated from the ledger, not appended. When a card is
  deleted its name disappears from history too. Otherwise deletion would leave
  the memory's title behind in a log file.
