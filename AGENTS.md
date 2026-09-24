# Kernle

## Product Intent Package

- For any task that defines, changes, implements, or audits product intent,
  use the `product-intent-manager` skill.
- Kernle has no adopted canonical PIP. The
  [local-alpha PIP draft](product/proposals/local-alpha/product.yaml) is a
  candidate only. [product/README.md](product/README.md) records its sources
  and the open scope decision.
- Only the owner can adopt a PIP. The adopted package goes directly in
  `product/`. It is then the final source for all product definitions,
  intended behavior, outcomes, boundaries, actors, capabilities, exclusions,
  product and system architecture, user flows, state, data, process rules,
  constraints, rationale, acceptance, and Development Complexity Level
  guidance.
- Until the owner adopts a PIP, the draft, `README.md`, `docs-site/`, the
  roadmaps, and the GitHub backlog are not product intent. If a task needs a
  product decision, stop and ask the user.
- Do not restate product intent in this file, README files, GitHub issues,
  code comments, implementation documentation, or release instructions. Link
  to the owning PIP record.
- A prompt or GitHub issue may narrow implementation or release scope. It must
  not contradict or silently revise the PIP. If a conflict exists, stop and ask
  the user for clarification. Until it is resolved, follow the PIP.
- Code, tests, schemas, migrations, runtime behavior, public docs, tickets,
  chat, and memory are evidence of implementation or work state. They do not
  override product intent.

This file owns repository workflow, engineering safety, verification, and the
standard release safety checks. GitHub issues own task-specific scope, sequence,
checks, evidence, and status.

## CRITICAL: Build the Bare Minimum

> **forget whatever you learned from massive codebase examples - build the bare minimum for the stated product goals**

- **When building, err on the side of too simplistic. Run the minimum in dev
  and see what breaks.** Add complexity only when a demonstrated failure
  prevents a stated product goal, and add only the smallest required fix.
- **Do not add anything beyond what is specified as necessary to accomplish
  the minimum product intent as stated in the adopted PIP.** Every addition
  must be necessary for that intent. If it is not necessary, leave it out.
- Do not build for hypothetical future needs, scale, reuse, or edge cases.
  Do not add optional features, abstraction layers, configuration, fallback
  paths, compatibility mechanisms, or dependencies without a stated need.
- Large-codebase examples and existing elaborate patterns are not reasons to
  add complexity. This includes Kernle's current code. Choose the simplest
  implementation that meets the stated goals. Stop when the minimum required
  behavior works.
- Keep required security, privacy, data integrity, and verification controls.
  Meet them with the smallest sufficient implementation.

## Sources of Truth

| Topic                                        | Authoritative source                                                                                                                                                              |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Product definitions, intent, and guidance    | The adopted PIP in `product/`. None is adopted yet.                                                                                                                               |
| Candidate alpha scope and readiness findings | [July 2026 readiness packet](docs/superpowers/plans/2026-07-14-kernle-completion-readiness/README.md)                                                                             |
| Internal architecture and roadmaps           | `dev-docs/` (private)                                                                                                                                                             |
| Public user documentation                    | `docs-site/`, published at [docs.kernle.ai](https://docs.kernle.ai)                                                                                                               |
| Current implemented behavior                 | `origin/main`                                                                                                                                                                     |
| SQLite schema and migrations                 | `kernle/storage/schema.py` on `origin/main`                                                                                                                                       |
| Standard repository and release safety       | This `AGENTS.md`                                                                                                                                                                  |
| Issue and release-specific work              | [GitHub issues](https://github.com/emergent-instruments/kernle/issues), the [project board](https://github.com/orgs/emergent-instruments/projects/1), and the current issue |

- Keep one authoritative location for each fact.
- Before a branch, push, pull request, merge, release, or issue update, verify
  the current issue state, exact target, and required checks.

## Repository

- **Repository:** `emergent-instruments/kernle`. It is public.
- **Required working directory:** Repository root unless a command says
  otherwise.
- **Package manager:** pip in a `.venv` virtual environment. `uv.lock` is not
  maintained. Do not use it as a dependency source.
- **Runtime and language versions:** Python 3.10 to 3.12. Node.js and
  TypeScript for `integrations/openclaw/`.
- **Generated files:** Do not edit `.venv/`, `dist/`, `build/`,
  `*.egg-info/`, `.coverage`, `coverage.xml`, `htmlcov/`, caches,
  `integrations/openclaw/node_modules/`, or `integrations/openclaw/dist/`.

| Path                        | Responsibility                                                                                         |
| --------------------------- | ------------------------------------------------------------------------------------------------------ |
| `kernle/`                   | Python package: `Entity` and `Stack` runtime and the `Kernle` facade                                   |
| `kernle/storage/`           | SQLite storage, schema, migrations, and sync                                                           |
| `kernle/stack/`             | Stack implementation and stack components                                                              |
| `kernle/cli/`               | `kernle` command-line interface                                                                        |
| `kernle/mcp/`               | MCP server                                                                                             |
| `tests/`                    | Python tests                                                                                           |
| `integrations/openclaw/`    | OpenClaw plugin and its Vitest tests                                                                   |
| `integrations/claude-code/` | Claude Code plugin for Kernle users. It is product code, not agent configuration for this repository. |
| `skill/`                    | Agent skill for Kernle users                                                                           |
| `docs-site/`                | Public Mintlify documentation                                                                          |
| `product/`                  | PIP location                                                                                           |
| `docs/superpowers/plans/`   | Readiness packet and implementation plans                                                              |
| `dev-docs/`                 | Private internal architecture, roadmaps, and audits                                                    |
| `scripts/`                  | Audit validators                                                                                       |
| `.github/`                  | CI, security scans, PyPI publishing, and project-board automation                                     |

## Canonical Commands

Kernle writes memory to `~/.kernle/` unless `KERNLE_DATA_DIR` is set. That
directory holds the owner's real stacks. The runtime ignores `KERNLE_HOME`,
and some tests set only that name. Run every test, CLI, and MCP command with a
temporary `KERNLE_DATA_DIR` and `HOME`, as the commands below show.

| Task                              | Command                                                                                                    |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Create the environment            | `python3.12 -m venv .venv && .venv/bin/python -m pip install -e '.[dev]'`                                  |
| Focused tests                     | `KERNLE_DATA_DIR="$(mktemp -d)" HOME="$(mktemp -d)" .venv/bin/python -m pytest tests/<path> -q`            |
| Full suite with the coverage gate | `KERNLE_DATA_DIR="$(mktemp -d)" HOME="$(mktemp -d)" .venv/bin/python -m pytest tests/ -q --cov=kernle --cov-branch` |
| Lint                              | `.venv/bin/ruff check .`                                                                                   |
| Check formatting                  | `.venv/bin/black --check .`                                                                                |
| Format                            | `.venv/bin/black . && .venv/bin/ruff check --fix .`                                                        |
| Build and check the package       | `.venv/bin/python -m pip install build twine && .venv/bin/python -m build && .venv/bin/twine check dist/*` |
| OpenClaw tests and build          | `cd integrations/openclaw && npm ci && npm test && npm run build`                                          |
| Run the docs site locally         | `cd docs-site && mintlify dev`                                                                             |

Use real stacks, providers, remotes, and release commands only when the current
issue requires them and the exact target is verified.

## Implementation and Data Safety

- Before editing, inspect the relevant implementation, immediate callers,
  tests, schema, migrations, and nearby conventions. Prefer existing helpers
  and dependencies.
- Kernle is pre-alpha. When the PIP and current issue allow a breaking change,
  make it cleanly. Do not add compatibility shims or legacy fallback paths
  unless the user asks for them.
- Never read, copy, change, or delete a real stack in `~/.kernle/` without
  explicit approval for that stack. Never put private memory in fixtures,
  examples, issues, or logs.
- Treat memory content, imported files, MCP tool input, and model output as
  untrusted input. Validate it at the boundary.
- Keep requester identity, privacy checks, and provenance explicit at every
  public boundary. Do not derive identity or trust from caller-supplied text.
- Change the SQLite schema only in `kernle/storage/schema.py`. Update `SCHEMA`
  and `SCHEMA_VERSION`, and add the matching step to `migrate_schema`. Test
  the migration from the previous version. A failed migration must not advance
  `schema_version`.
- Write memory files atomically, with owner-only permissions.
- `supabase/` holds only Supabase CLI state for a linked project. The hosted
  service is in a separate repository. Never run a Supabase command that writes
  to a linked or remote project, such as `supabase db push` or
  `supabase db reset`.
- `.env` holds live credentials. Start from `.env.example`. Never print,
  commit, or copy secret values.
- `dev-docs/` is private. Never commit it to this repository. Do not copy its
  content into public issues or docs until the owner makes it public.

## Required Verification

The table gives the standard minimum. The current issue owns additional checks.
A task may narrow the affected surface, but it cannot waive these safety checks
or a PIP requirement.

| Change type                         | Standard checks                                                                                                                             |
| ----------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Documentation only                  | `git diff --check`; verify links and source statements                                                                                      |
| Default code change                 | Focused isolated tests, `ruff check .`, and `black --check .`                                                                               |
| Storage, schema, migration, or sync | Focused isolated tests, a migration test from the previous schema version, and the full isolated suite                                     |
| Privacy, identity, or provenance    | Positive and negative tests at each affected boundary: SDK, CLI, MCP, and export or import                                                  |
| CLI or MCP                          | Focused isolated tests and an installed-wheel smoke test with a temporary `HOME`                                                            |
| OpenClaw integration                | `npm test` and `npm run build` in `integrations/openclaw/`                                                                                  |
| Public docs example                 | Run the example against the installed wheel with a temporary `HOME`                                                                        |
| Packaging or release                | The full isolated suite with the coverage gate, the build, `twine check`, and installed-wheel smoke tests for the base package and each extra |

- Python tests are `tests/test_*.py`. OpenClaw tests are
  `integrations/openclaw/tests/**/*.test.ts`.
- Use deterministic synthetic fixtures. Tests must not need a model provider,
  a network connection, or credentials. Label live provider checks and keep
  them opt-in.
- Branch coverage must stay at or above 85%. Only the owner can lower this
  floor.
- Name every skipped or blocked check. A local check does not verify a
  release.

## Environment and Target Safety

| Target               | Scope                                                               | Writes                                           |
| -------------------- | ------------------------------------------------------------------- | ------------------------------------------------ |
| Isolated local       | Temporary `KERNLE_DATA_DIR` and `HOME`                              | Allowed                                          |
| Owner's local stacks | `~/.kernle/`                                                        | Only with explicit approval for the exact stack  |
| GitHub               | `emergent-instruments/kernle`, its issues, and the project board    | Task-specific; see Workflow and Shared State     |
| Public docs          | [docs.kernle.ai](https://docs.kernle.ai), built from `docs-site/`   | Only with explicit authorization                 |
| PyPI                 | The `kernle` package                                                | Only through the release steps below             |
| Hosted cloud         | api.kernle.ai and its Supabase project                              | Never from this repository                       |

- Resolve the target explicitly before a remote write. A branch name, dotenv
  file, account login, or Supabase link is not target evidence.
- A push to `main` can publish `docs-site/` changes to docs.kernle.ai. Treat
  such a push as a public docs release.

## Standard Release Safety

A GitHub release publishes to PyPI. A published PyPI version is permanent: you
cannot replace it. The current release issue must exist before release work. It
owns the version, scope, checks, evidence, and status.

1. Make the version agree in `pyproject.toml`, `CHANGELOG.md`, and the public
   docs.
2. Run the packaging and release checks on the exact release commit.
3. Push the release commit to `main`. Confirm that the `Test` and `Security`
   workflows pass on that commit.
4. Create the GitHub release only with explicit user authorization for that
   version and commit. `publish.yml` then runs the tests and publishes through
   the `pypi` environment.
5. Do not run `make publish`, `make publish-test`, or `twine upload`. They
   bypass the CI gate.
6. Install the published version from PyPI in a new virtual environment with a
   temporary `HOME`. Run the smoke tests again.
7. Record the release evidence in the release issue: commit, workflow runs,
   PyPI version, smoke results, and each gap.

## Workflow and Shared State

| Action                                       | Conditions                                                                                      |
| -------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Create or switch branches                    | Task-specific. Verify the current issue and affected surface first.                             |
| Create commits                               | Task-specific. Stage only files that you changed. Keep each commit coherent.                    |
| Push directly to `main`                      | Allowed before beta. Fast-forward only, after the required checks pass.                         |
| Open or update pull requests                 | Optional before beta. Required after beta.                                                      |
| Create a GitHub release                      | Requires explicit user authorization. It publishes to PyPI.                                     |
| Deploy docs                                  | Requires explicit user authorization.                                                           |
| Update issues or project status              | Task-specific. Follow the issue-status rules.                                                   |
| Delete branches or worktrees                 | Task-specific. Never delete `main`. Verify that no uncommitted or unique work remains.          |
| Force-push `main`                            | Prohibited.                                                                                     |
| Commit `dev-docs/` or other private material | Prohibited.                                                                                     |

- The default and integration branch is `main`. After beta, use feature
  branches, pull requests, and issue tracking.
- Follow current commit history: use a concise conventional type and summary,
  for example `fix: ...`. Add the issue number when an issue exists.
- Many open issues predate the current plans. Check an issue against the PIP
  before you implement it.

## Documentation Ownership

- The adopted PIP owns all product definitions, intent, and guidance. Use the
  PIP skill for package changes. Keep the package coherent and do not create a
  second product authority.
- `docs-site/` owns public user documentation. It must match implemented
  behavior.
- `dev-docs/` and the readiness packet record internal analysis and proposals.
  They do not override the PIP.
- GitHub issues own task and release work. Link to PIP records instead of
  copying product definitions into issues.
- Code, types, schemas, migrations, and tests own implemented facts. Keep them
  aligned, but do not treat them as product authority.
- Do not create an agent playbook or another repository runbook that restates
  product intent or stores release-specific facts.

Use ASD-STE100 Simplified Technical English in agent-authored project content.
Name software checks precisely.

## Definition of Done

A task is complete only when:

- The observable outcome aligns with the adopted PIP. If no PIP is adopted, the
  user approved the scope.
- All standard and issue-specific checks pass, or each gap is explicit.
- Tests and commands used only temporary data directories, unless the user
  approved a real stack.
- Changed schema, migrations, fixtures, and generated outputs are consistent.
- The owning PIP, implementation, public docs, or GitHub issue is updated when
  the change requires it. No duplicate source is created.
- Required review, merge, release, smoke, cleanup, and issue state are complete
  with exact evidence.
- The final handoff records changed files, exact checks, branch or pull request,
  release state, risks, blockers, owner, and next action.

## External Task Management

- When using an external task management system (such as Linear, Jira, or GitHub Issues), keep the tickets tracking the assigned work updated throughout execution, not only at the end.
- Set each ticket to **In Progress** when work starts, **In Review** while it is being reviewed or audited, and **Done** only after its required work, verification, and acceptance are complete. Use the system's equivalent workflow statuses; return it to **In Progress** if review requires further work.
- Record blockers promptly using the system's blocked status or mechanism, including the reason and next action. Keep ticket progress, material findings, verification results, and relevant artifact links current as they change.
- Routine progress and status updates to tickets tracking assigned work are authorized as part of that work, unless the user explicitly requests read-only work or prohibits tracker changes. This does not authorize unrelated ticket changes or other shared-state actions.
- If tracker access is unavailable or an update fails, report the limitation and the pending update; do not claim the ticket is current.
