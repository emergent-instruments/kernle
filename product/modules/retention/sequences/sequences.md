# Retention sequences

Retention owns forgetting: scheduled expiry, requests to forget, idle space
deletion, and making deletion irrecoverable from Kernle's files. It owns no
tables. It deletes rows owned by other modules through their operations, or
through the explicit deletions named below. Rust module `retention` owns these
processes. Parameters are in [settings.yaml](../contracts/settings.yaml).

Every space database connection opens with `PRAGMA secure_delete = ON`, so
deleted rows are overwritten in place, not left in free pages.

## Public boundary

| Operation | Callers | Result and guarantees |
| --- | --- | --- |
| [RetentionSweep](#seq-060-retention-sweep) | consolidation SEQ-031 (each sleep) and SEQ-030 (daily retention run) | Applies every window in settings and RULE-040. When it returns, deleted content is absent from the space's database, WAL, search index, files, projected skills, and changelog |
| [Forget on request](#seq-061-forget-on-request) | `kernle forget` (interactive owner), agent `forget` tool | Owner: previewed cascade, then complete deletion. Agent: deletes one unprotected card, and holds pinned cards and guardrails for the owner |
| [Idle space deletion](#seq-062-idle-space-deletion) | Daily retention run over the registry | Deletes a whole space after `space_idle_days` without capture or recall activity, and shows the date in status from `space_warning_days` before |

## SEQ-060 Retention sweep

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
  participant R as ARCH-008 retention sweep
  participant DB as ARCH-009 space kernle.db
  participant PDB as ARCH-009 other spaces' kernle.db
  participant FS as ARCH-009 space files

  Note over R: Runs while the caller holds the space's run guard (consolidation SEQ-030)
  R->>DB: Delete expired events
  Note right of DB: DELETE · DATA-002 event<br/>ACCESS · DATA-002 event [I2] event_expires_idx<br/>GUARD · expires_at ≤ now
  R->>DB: Find expired episodes
  Note right of DB: READ · DATA-004 episode<br/>ACCESS · DATA-004 episode [I1] episode_expires_idx<br/>GUARD · expires_at ≤ now
  loop Each expired episode
    R->>DB: Check citations in this space
    Note right of DB: READ · DATA-009 card_source [I1] card_source_source_idx<br/>JOIN · DATA-007 card status active or dormant
    R->>PDB: Check citations from the personal space (or from project spaces when sweeping the personal space)
    Note right of PDB: READ · DATA-009 card_source [I1] with source_space_id = this space
    alt Cited by an active or dormant card
      Note over R: KEEP · the gist that justifies a live memory stays
    else Not cited by a live card
      R->>DB: Delete the episode and the citations held by superseded cards
      Note right of DB: DELETE · DATA-004 episode · KEY · id<br/>DELETE · DATA-002 event · KEY · episode_id (any remaining)<br/>DELETE · DATA-009 card_source rows citing it, in this space and others
    end
  end
  loop Each card (DATA-007)
    R->>R: Decide with RULE-040
    R->>DB: UPDATE · DATA-007 card.strength (retention-owned)
    alt DELETE
      R->>R: memory ApplyCardOp(delete, reason) (memory SEQ-040)
      R->>DB: DELETE · DATA-012 recall_log · ACCESS · DATA-012 recall_log [I2] recall_log_card_idx
    else DORMANT or ACTIVE with a status change
      R->>R: memory ApplyCardOp(set_status, dormant or active) (memory SEQ-040)
    end
  end
  R->>R: consolidation expire held changes (consolidation SEQ-032, expiry branch)
  R->>DB: Delete aged records
  Note right of DB: DELETE · DATA-006 card_op · GUARD · decided_at older than card_op_retention_days<br/>DELETE · DATA-012 recall_log · GUARD · expires_at ≤ now
  R->>FS: Remove changelog files older than changelog_retention_days
  opt Any card was deleted in this sweep
    R->>R: recall MaintainIndex, synchronous (recall SEQ-055)
    R->>DB: Purge deleted text from the full-text index and the WAL
    Note right of DB: INSERT INTO card_fts(card_fts) VALUES('optimize')<br/>PRAGMA wal_checkpoint(TRUNCATE)
    R->>R: memory WriteChangelog(affected dates) (memory SEQ-042)
  end
```

### Current rationale

- Expired events are deleted on schedule even when their episode is kept.
  Raw transcript excerpts are the most sensitive data, and the episode's
  summary is what justifies a live memory.
- A dormant card still protects its episodes. Dormancy is the warning stage
  before deletion, and its evidence must survive until the card itself goes.
- The full-text index is optimized and the WAL truncated after deletions.
  SQLite full-text segments and write-ahead logs otherwise keep deleted text
  until a later rewrite, and "deleted" must mean gone from Kernle's files.

## SEQ-061 Forget on request

The owner runs `kernle forget <card name, id, or search text>` in an
interactive terminal, or the agent calls the `forget` tool.

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
  participant Q as Owner CLI or agent tool
  participant D as ARCH-008 kernled
  participant DB as ARCH-009 space kernle.db (and cited spaces)

  Q->>D: Forget(card reference)
  D->>D: spaces ResolveSpace (spaces SEQ-003) and find the card
  Note right of D: Owner: exact name or id, or a RULE-030 search including dormant cards for the owner to pick<br/>Agent: exact name or id in the readable spaces only
  alt Card not found
    D-->>Q: not found
  else Requester is the agent
    alt Card is pinned or kind = guardrail
      D->>DB: Hold the deletion for the owner
      Note right of DB: INSERT · DATA-006 card_op op 'delete', policy 'review',<br/>state 'pending_review', reason 'agent requested forget'
      D-->>Q: held_for_owner
    else Unprotected card
      D->>D: memory ApplyCardOp(delete, reason 'agent request') (memory SEQ-040)
      D->>DB: DELETE · DATA-012 recall_log [I2] rows for the card
      D->>D: recall MaintainIndex, synchronous (recall SEQ-055)
      D-->>Q: forgotten
    end
  else Requester is the owner
    alt stdin or stdout is not an interactive terminal
      D-->>Q: Refused · forget needs an interactive terminal · nothing deleted
    else Interactive
      D->>DB: Build the cascade preview
      Note right of DB: READ · the card and all DATA-008 versions<br/>DATA-009 card_source: merged_from and supersedes predecessors, followed recursively<br/>evidence episodes (DATA-004) and their events (DATA-002) that no other live card cites<br/>DATA-012 recall_log rows · projected skill folder<br/>Optional: other live cards citing this card or sharing all of its evidence
      D-->>Q: Preview (FLOW-006), including that runtimes keep their own transcripts
      Q->>D: Confirm, with any optional cards selected · or Cancel
      alt Cancel
        D-->>Q: Nothing deleted
      else Confirm
        loop Each card in the confirmed set
          D->>D: memory ApplyCardOp(delete, reason 'owner forget') (memory SEQ-040)
        end
        D->>DB: Delete the cascaded evidence and usage
        Note right of DB: DELETE · DATA-004 episodes and DATA-002 events from the preview<br/>DELETE · DATA-009 card_source rows in any space citing them<br/>DELETE · DATA-012 recall_log rows for the deleted cards<br/>TRANSACTION · per space database
        D->>D: recall MaintainIndex, synchronous (recall SEQ-055)
        D->>DB: INSERT INTO card_fts(card_fts) VALUES('optimize') · PRAGMA wal_checkpoint(TRUNCATE)
        D->>D: memory WriteChangelog(affected dates) (memory SEQ-042)
        D-->>Q: Forgotten: counts of cards, versions, episodes, and events removed
      end
    end
  end
```

### Current rationale

- The owner sees the cascade before anything is deleted, because forgetting
  is irreversible and the evidence behind a memory may be what the owner
  actually wants gone.
- Agents cannot delete pinned cards or guardrails. Instructions injected into
  an agent could otherwise make it delete the very memories that guard
  against them. The owner decides those deletions through review.
- An agent's forget deletes only the card, without cascade. The owner never
  saw what else would go, and agent deletions should be the minimum that
  honors the request.
- Kernle never deletes the runtimes' own transcripts, and the preview says
  so. Those files belong to the runtime, so complete forgetting also needs the
  runtime's own controls.

## SEQ-062 Idle space deletion

Runs once a day inside the scheduler's retention run, over the registry.

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
  participant R as ARCH-008 retention run
  participant REG as ARCH-009 registry.db
  participant DB as ARCH-009 other spaces' kernle.db
  participant FS as ARCH-009 space directory
  participant SK as ARCH-010 ~/.claude/skills

  R->>REG: Read every space's last activity
  Note right of REG: READ · DATA-001 space.last_active_at<br/>deletion_date = last_active_at + space_idle_days<br/>kernle status shows deletion_date once now ≥ deletion_date - space_warning_days
  loop Each space with now ≥ deletion_date
    R->>R: Wait for and take the space's run guard (consolidation SEQ-030)
    R->>SK: Remove the space's projected skill folders (those carrying .kernle-generated)
    R->>DB: Delete citations of this space's episodes held by other spaces
    Note right of DB: DELETE · DATA-009 card_source<br/>KEY · source_space_id = this space
    R->>FS: Remove spaces/{id}/ recursively
    Note right of FS: cards, skills, changelog, kernle.db, and its WAL and SHM files
    R->>REG: DELETE · DATA-001 space · KEY · id
    Note over R: The personal space is recreated empty on its next use (spaces SEQ-003)
  end
```

### Current rationale

- Idle deletion removes the whole directory rather than sweeping rows. A
  year without any use means the owner no longer works there, and whole-file
  removal leaves nothing behind in database pages.
- The deletion date appears in status a month ahead, because removal is
  irreversible and the only thing that prevents it is using the space.
