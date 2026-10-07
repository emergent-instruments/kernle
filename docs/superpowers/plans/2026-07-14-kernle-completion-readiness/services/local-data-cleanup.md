# Service: Local Test-Data Cleanup

## Purpose

Recover from existing tests that wrote test-named rows to the real `~/.kernle/memories.db` because they set ignored `KERNLE_HOME` instead of `KERNLE_DATA_DIR`.

## Required Config and Secrets

- No secrets.
- Database: `~/.kernle/memories.db` only after the owner confirms it is the affected file.
- Owner-approved exact stack IDs and time bounds for disposable rows.

## Human Setup Steps

1. Stop all Kernle processes that may write the database.
2. Make a timestamped byte-for-byte backup and verify it opens read-only.
3. Produce a read-only table-by-table inventory for candidate stack IDs, including row count and min/max timestamps.
4. Owner explicitly marks each stack/table/time range disposable. Never infer deletion from a `test-` prefix alone.
5. Generate exact delete statements and pre/post expected counts for review.
6. Execute once in a transaction, verify invariants, and keep the backup until the owner accepts the result.

## Allowed and Forbidden Behavior

- Allowed without further approval: read-only inventory and cleanup-plan drafting.
- Forbidden without explicit approval: delete/update, broad wildcard cleanup, database replacement, `VACUUM`, or backup removal.
- Do not use `supabase db reset` or any remote command.

## Test and Smoke Requirements

- Backup opens with `sqlite3 -readonly` and passes `PRAGMA integrity_check`.
- Planned delete counts equal owner-approved candidate counts before execution.
- Transaction rolls back on any count mismatch.
- Post-cleanup integrity, non-candidate row counts, and owner stack samples are unchanged.

## Current Blockers

- No pre-audit snapshot exists, so exact attribution by timestamp alone is unsafe.
- Test-prefixed data predates this audit and may include intentional artifacts.
- Owner has not authorized any deletion.

