# Bundle: Atomic Write Integrity

Type: worker/backend

Owns: Stack/Core propagation of backend write results and post-save hook dispatch. It does not define new privacy or provenance policy.

Status: blocked on decision 0005. Repair only affected operations that remain in alpha; if goal/drive/relationship paths are removed, close this bundle as not applicable instead of fixing code that Bundle 06 deletes.

## Assignment Context

Before implementation, read [../assignment-context.md](../assignment-context.md). Then use this bundle's Services, Libraries, API Guidance, Tests, Raw Fixtures, and Done sections as the claim scope.

## Product Function

Callers and plugins observe success only when a goal, drive, or relationship update actually committed.

## Services

- [Local development](../services/local-development.md).

## Libraries

- Existing pytest and in-memory/fake storage patterns; no dependency needed.

## API / Components

- Preserve backend boolean/conflict outcomes.
- Dispatch `on_save` only after success.
- Make only retained goal/drive/relationship semantics consistent; otherwise cover deletion/quarantine in Bundle 06.

## Implementation Notes

- First reproduce false/conflict returns with a real narrow fake or SQLite conflict.
- Do not broaden this into API consolidation.

## Unit Tests

- False backend result returns false and dispatches no hook.
- Optimistic-version conflict is observable and leaves the record unchanged.
- Success dispatches exactly once.

## Integration Tests

- Real SQLite conflict/success paths for retained atomic updates.

## E2E Tests

- Not required; behavior is covered at the public Stack contract.

## Raw Fixtures

- Generated records; no owner material required.

## Done

- Every retained atomic update reports the backend result honestly.
- Plugins never receive a save event for a failed write.
- Focused and affected integration tests pass.
