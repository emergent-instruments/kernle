# Kernle

**Stratified, appraisal-driven memory for AI agents.**

Kernle gives agents a memory of who they are and what they have lived through:
raw captures, episodes, notes, beliefs, values, goals, drives, relationships,
playbooks, guardrails, and a self-narrative. It captures experience
automatically from agent sessions and through deliberate calls, works out what
mattered, consolidates it while the agent is idle, and brings the right part
back at the right moment. Agents stop repeating mistakes, and the person who
runs them can see, correct, and truly delete every memory.

Kernle's "emotion" is a computed appraisal, not a reported feeling. Each
episode is appraised from observable signals: stakes, surprise, outcome, the
person's feedback, effort, novelty, and recurrence. The appraisal decides how
the experience is filed, how highly it ranks when recalled, and how long it
lasts:

| Appraisal | Becomes |
| --- | --- |
| Alarm | Guardrail, checked before risky actions |
| Frustration | Learned belief and a lesson on the episode |
| Correction | Preference belief, or a guardrail |
| Relief | Playbook for a repeatable procedure |
| Surprise | Revision of the contradicted belief |
| Approval | Reinforcement of what was followed |
| Curiosity | Exploration goal |
| Boredom | Nothing new; it reinforces what is known or fades |

Kernle 1.0 is a Rust engine with a per-user daemon, shipped as a Python
package (`pip install kernle`) that includes the native `kernle` command-line
tool. It works with Claude Code, Codex CLI, and OpenClaw, runs on macOS and
Linux machines the owner controls, and can optionally sync a stack to the
Kernle cloud service.

## Where to read

The [Ready Intent Package](product/product.ready) in `product/` is the
canonical description of Kernle 1.0: one `.ready` record per module, journey,
sequence, rule, entity, and operation. Validate it with
`npx @readyroom/ready validate .`. Start with the product record, then the
modules:

- [stacks](product/stacks/module.ready): home, stacks, owner-only access, model provider, and the daemon
- [agents](product/agents/module.ready): hooks, the MCP tool server, the OpenClaw plugin, and setup
- [capture](product/capture/module.ready): raw captures, transcript ingestion, imports, and seeding
- [memory](product/memory/module.ready): every memory kind, versions, provenance, and export
- [appraisal](product/appraisal/module.ready): the computed emotion
- [consolidation](product/consolidation/module.ready): processing, sleep, suggestions, and review
- [identity](product/identity/module.ready): values, self-narrative, epochs, and summaries
- [trust](product/trust/module.ready): source trust tiers, the input gate, and entity trust
- [recall](product/recall/module.ready): load, search, cues, gut checks, and checkpoints
- [retention](product/retention/module.ready): decay, forgetting, recovery, and real deletion
- [health](product/health/module.ready): anxiety, doctor, and metamemory
- [sync](product/sync/module.ready): cloud account, push and pull, and conflicts

The [review and proposal](docs/proposals/2026-10-05-kernle-review-and-proposal.md)
explains what the earlier Python implementation intended, what it actually
did, and the reasoning behind this design.

## History

The earlier Python implementation (v0.15, "stratified memory for synthetic
intelligences") remains in this repository's Git history. Its last state is
commit `499af9f`.

## License

MIT
