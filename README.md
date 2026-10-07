# Kernle

**An appraisal-driven experience layer for AI agents.**

Kernle captures what happens in an agent's sessions, works out what mattered,
consolidates it while the agent is idle, and brings the right memory back at
the right moment. Agents stop repeating mistakes and start compounding skill,
without spending any effort on memory upkeep.

Kernle's "emotion" is a computed appraisal, not a reported feeling. Each
episode of work is appraised from observable signals: surprise, outcome,
stakes, user corrections, effort, novelty, and recurrence. That appraisal
decides how the experience is filed:

| Appraisal | Becomes |
| --- | --- |
| Alarm | Guardrail, checked before risky actions |
| Frustration | Lesson |
| Correction | Preference |
| Relief | Skill |
| Surprise | Fact revision |
| Boredom | Nothing new; it reinforces what is already known or fades |

It also decides how long the memory lasts, when it comes back, and when its
urgency fades.

Kernle 1.0 is a single Rust binary that runs locally on machines the owner
controls. It works with Claude Code, Codex CLI, and OpenClaw. Memory stays
private, lives in plain markdown files, traces back to its sources, and can be
deleted completely.

## Where to read

- [Product intent package](product/product.yaml) is the canonical description
  of Kernle 1.0. Start with the [stack context](product/architecture/stack-context.md),
  then the modules:
  - [spaces](product/modules/spaces/sequences/sequences.md): home, spaces, owner-only access, and the daemon
  - [capture](product/modules/capture/sequences/sequences.md): how session activity becomes events
  - [appraisal](product/modules/appraisal/behavior/rules.md): the computed emotion
  - [consolidation](product/modules/consolidation/sequences/sequences.md): sleep, filing, policy, and review
  - [memory](product/modules/memory/sequences/sequences.md): memory cards and skills as files
  - [recall](product/modules/recall/sequences/sequences.md): session index, cue injection, and gut checks
  - [retention](product/modules/retention/sequences/sequences.md): expiry and real deletion
- [Review and proposal](docs/proposals/2026-10-05-kernle-review-and-proposal.md)
  explains why Kernle was redesigned. It covers what the earlier Python
  implementation intended, what it actually did, and the reasoning behind the
  new approach.

## History

The earlier Python implementation (v0.15, "stratified memory for synthetic
intelligences") remains in this repository's Git history. Its last state is
commit `499af9f`.

## License

MIT
