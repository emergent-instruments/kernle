# Kernle Product Intent Package

The [local-alpha PIP draft](proposals/local-alpha/product.yaml) is a coherent,
reviewable interpretation of the July 14 documentation. It is isolated in
`proposals/local-alpha/` because the source documents explicitly leave its
release scope for the owner to choose. It is not adopted canonical intent.

The pending scope choice is whether Kernle's PIP describes this minimal local
alpha, or the full cognitive memory product with a narrower alpha release
boundary. The current user request authorizes reconstruction, but does not
explicitly settle that difference between the source documents.

The draft contains the product record, physical stack and trust boundaries,
user-visible flows, memory/provenance model, and runtime sequences. Acceptance
is inline with the capabilities. It contains no implementation-readiness
claims, delivery tasks, or audit-history records.

## Source chronology and interpretation

| Source | Date evidence | Treatment |
| --- | --- | --- |
| [README](../README.md) | Filesystem last edit February 13, 2026 | Product background; quickstart and feature/maturity claims are not current intent by themselves. |
| [Core processes](../dev-docs/reference/core-processes.md) and [support processes](../dev-docs/reference/support-processes.md) | Nested audit Git history: core February 27; support February 24, 2026 | Historical cognition and interface detail. Later contradictory architecture and release guidance takes precedence in this reconstruction. |
| [Protocol/storage architecture](../dev-docs/protocol-storage-architecture.md) and [framework roadmap](../dev-docs/framework-catch-up-roadmap.md) | Internal date and filesystem edits May 4, 2026; untracked in the nested audit repository | Newer protocol-first direction: common policy, immutable provenance, storage adapters, explicit remotes, pending memory and trust. |
| [Protocol refactor roadmap](../dev-docs/ROADMAP.md) | Internal date and filesystem edit May 5, 2026; untracked in the nested audit repository | Adopts the May architecture and prioritizes privacy, provenance, API consolidation and conformance before broader features. |
| [July packet](../docs/superpowers/plans/2026-07-14-kernle-completion-readiness/README.md), [alpha simplification](../docs/superpowers/plans/2026-07-14-kernle-completion-readiness/alpha-simplification.md), and [surface guidance](../docs/superpowers/plans/2026-07-14-kernle-completion-readiness/ui-guidance.md) | Added July 14, 2026; filesystem edits through 20:39 local time; untracked with no Git history | Most recent product-cut recommendation. It supplies the candidate alpha boundary and concrete safety/recovery requirements, while explicitly leaving adoption to the owner. |

Filesystem modification time is supporting evidence, not an approval signal.
The core-process file's February 26 filesystem time predates its February 27
Git edit, illustrating why Git history and the substance of each change matter.
The `audits/` to `dev-docs/` working-tree move has not been committed; this task
does not alter that move or the existing source documents.

The file `dev-docs/schema.yaml` is an audit-report schema, not Kernle's memory
schema. Runtime names and storage fields in the PIP were grounded in the Python
sources. Existing code and tests were used to find owners and hazards, not to
turn current defects or old behavior into requirements. No implementation
alignment or application test result is asserted by this documentation work.

## Material interpretation choices in the candidate

- The July recommendation supplies Entity + Stack, MCP, raw plus episode,
  explicit promotion, owner-only local identity, and one SQLite database per
  stack. The draft uses those together as one consistent candidate.
- May's protocol ownership and common privacy/provenance boundary remain valid
  within that alpha. Multiple storage adapters, remotes, sharing, pending
  memory, skills and broader cognition do not become alpha commitments merely
  because they appear in May's longer-term architecture.
- February's autonomous model pipeline, broad CLI/MCP write and lifecycle
  surfaces, suggestion reviews, checkpoints, and cloud behavior are not mixed
  into the model-free July alpha.
- The source leaves destructive lifecycle and restore collision rules open.
  The candidate excludes public in-place edits, deletion/erasure, provenance
  repair, merge restore, owner transfer and automatic legacy conversion.
  Restore accepts an empty destination with the same trusted owner and stack
  binding. These conservative boundaries require review with the scope choice;
  they are not represented as historical owner decisions.
- The candidate specifies host-persisted owner/stack binding, atomic request
  reconciliation, and raw-origin audit retrieval to make the recommended
  integrity contract internally complete. These are reconstruction details,
  not claims that the runtime already enforces them.

The [July owner-input document](../docs/superpowers/plans/2026-07-14-kernle-completion-readiness/source-docs/inputs-required.md)
explicitly asks the owner to approve or replace the alpha cut. No adopted
brief or decision files settling that scope were found. Once the scope is
settled, the selected coherent package can occupy `product/` directly, and this
reconstruction note remains outside the canonical package.
