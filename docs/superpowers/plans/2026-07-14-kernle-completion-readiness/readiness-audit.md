# Kernle Completion Readiness Audit

Date: 2026-07-14  
Repository: `/Users/seanhart/emergent-instruments/kernle`  
Review mode: read-only product, architecture, security, testing, CI, docs, git, and public GitHub audit

## Outcome

**RED for autonomous feature implementation. AMBER for bounded stabilization.**

The codebase has a large, fast regression suite and coherent local-memory concepts. It does not yet have one trustworthy project authority, one canonical API, executable minimal workflows, enforceable privacy/provenance semantics, or approved workflow/security/MCP evidence. Those are inputs to the alpha implementation, not details agents should invent. Broader semantic/model qualification is required only for later claims.

## Findings

### Critical

1. **Fleet authority describes the wrong product.**

   `AGENTS.md:137-159` and `CLAUDE.md:137-159` describe Supabase/Postgres, Next.js, Expo/React Native, and a production deployment. The actual package is Python/SQLite with CLI and MCP entry points (`pyproject.toml:5-38,67-69`) plus a TypeScript OpenClaw integration. An obedient agent could build or deploy the wrong architecture.

2. **The current source-of-truth move is incomplete.**

   The root tracks a private submodule at `audits/` (`.gitmodules:1-4`), but the worktree deletes it and adds untracked `dev-docs/`. `dev-docs/.git` still points at `.git/modules/audits`, whose configured worktree is the deleted path. The main internal roadmaps are therefore untracked and the nested checkout cannot run normal git commands. The root is also 23 commits ahead of `origin/main` with other user-owned edits.

3. **Published private-by-default behavior is not enforced at public boundaries.**

   Public docs promise contextual privacy and consent (`docs-site/concepts/privacy.mdx:87-200,224-278`). SQLite can filter by `requesting_entity`, but treats `None` as unrestricted self-access (`kernle/storage/sqlite.py:645-678`). `Entity.search/load` accept only an unrelated context string (`kernle/entity.py:988-1007`); Stack getters/search/load omit requester identity (`kernle/stack/sqlite_stack.py:1230-1447`); dump reads storage directly (`kernle/core/serializers.py:15-100`). CLI, MCP, plugins, import/export, sync, and serialization therefore cannot consistently express who is asking.

4. **A supported OpenClaw path can lose a memory write.**

   The hook captures a native memory-file write through `bridge.raw`, ignores the returned success boolean, and always blocks the original write (`integrations/openclaw/src/hooks/before-tool-call.ts:43-57`; `services/kernle-bridge.ts:52-57`). It also resolves this hook's stack once with an empty session context (`src/index.ts:28-31`) while session start/end resolve the actual context. This can silently discard content or route it to a different stack.

5. **Documented basic capture is inconsistent with strict production behavior.**

   `Kernle` defaults to strict provenance and immediately activates its Stack (`kernle/core/kernle_class.py:86-133`). Active structured writes require a real derived reference (`kernle/stack/sqlite_stack.py:719-758`). The CLI creates default `Kernle` (`kernle/cli/__main__.py:2244-2252`), while the quickstart records notes/episodes without provenance (`docs-site/quickstart.mdx:40-54`). MCP integration tests use `strict=False` (`tests/test_mcp_integration.py:32-41`), masking the shipped path.

6. **Default multi-stack storage can overwrite or sync another stack's records.**

   Every default SQLiteStorage resolves to the same `~/.kernle/memories.db` (`kernle/storage/sqlite.py:415-425`), but memory IDs are globally primary-keyed rather than `(stack_id, id)` (`kernle/storage/schema.py:76-79` and repeated tables). Saves use `INSERT OR REPLACE`, so the same ID in stack B replaces stack A (`kernle/storage/episodes_crud.py:62-118`). `sync_queue`, `sync_meta`, and conflicts are also unscoped; queue uniqueness is only `(table_name, record_id)` (`kernle/storage/schema.py:624-670`) and every stack reads the same pending queue (`kernle/storage/sync_engine.py:207-241`). This is a deterministic isolation/data-loss risk.

7. **The test suite is not safely isolated from real local memory.**

   Runtime path resolution honors `KERNLE_DATA_DIR`, not `KERNLE_HOME` (`kernle/utils.py:13-18`), but MCP and storage tests set the ignored name (`tests/test_mcp_integration.py:32-47`; `tests/test_storage_discovery.py:99-123`). During this audit's full-suite window, `~/.kernle/memories.db` was modified and at least 65 rows dated today appeared across test-named episode/note/raw stacks. There was no pre-run database snapshot, so exact attribution may be broader or narrower. No cleanup was attempted. Full-suite execution must stop until every test process gets a temporary `KERNLE_DATA_DIR` and `HOME` plus a guard against the real path. Any cleanup must follow the owner-approved process in [services/local-data-cleanup.md](services/local-data-cleanup.md).

### High

8. **Provenance is neither immutable nor an authentic trust boundary.**

   Ordinary storage saves accept caller-supplied provenance and use replacement/upsert paths (`kernle/storage/beliefs_crud.py:58-108`, `episodes_crud.py:62-114`, `notes_crud.py:57-99`, `goals_crud.py:58-99`). Some updates overwrite provenance. Stack also trusts caller text starting with `kernle:` (`kernle/stack/sqlite_stack.py:698-717`), while public Entity methods can copy arbitrary `source` into `source_entity`. Trusted actor identity must come from an internal request context or capability, not a forgeable string.

9. **Migration failures can be hidden by a current schema version.**

   Database initialization runs best-effort migrations and then unconditionally writes the current version (`kernle/storage/schema.py:824-857`). A partially migrated database can appear current. The suite has hand-built in-memory legacy schemas, but no sanitized real database from each supported release.

10. **Atomic write outcomes are misreported.**

   Stack goal updates ignore the backend result; drive and relationship updates dispatch save hooks even when the backend returns false (`kernle/stack/sqlite_stack.py:847-881`). This can tell plugins/callers a failed write succeeded.

11. **The adapter contract is broader than its validation.**

   Storage composition exposes many method families, while the factory validates only four methods (`kernle/storage/factory.py:43-49`) and several protocol defaults silently no-op. Stack construction immediately assumes additional capabilities. A third-party adapter can pass creation and fail later.

12. **Public API and next-release truth are unresolved.**

    Both `Kernle` and `Entity` are public (`kernle/__init__.py:7-19`). `Kernle` is an 18-mixin facade but recommends Entity. `dev-docs/ROADMAP.md:37-62` explicitly leaves the canonical API and version meaning open. Public roadmap text still calls v0.14/v0.15 work planned (`docs-site/design/roadmap.mdx:237-240`) while the package is 0.15.0, and `CHANGELOG.md` stops at 0.13.05.

13. **The public Python quickstart cannot execute as written.**

    It constructs `Entity()` without required `core_id`, imports nonexistent `SQLiteStack`, and calls an incompatible constructor (`docs-site/quickstart.mdx:79-118`; `kernle/entity.py:406-414`; `kernle/stack/__init__.py:1-9`). README also points to missing `dev/README.md` and root `ROADMAP.md` (`README.md:85-100,173`).

14. **Build/configuration surfaces have stale product remnants.**

    Make targets and a CI workflow target nonexistent `backend/` and `web/` directories (`Makefile:81-94`; `.github/workflows/backend.yml:1-50`). `.env.example` is mostly Supabase and omits actual Kernle development/test variables. The OpenAI adapter advertises `kernle[openai]`, but `pyproject.toml:40-65` defines no OpenAI extra. `uv.lock:758-816` still identifies Kernle 0.13.3. Auth/sync code imports undeclared `httpx`, and the base package installs a broken optional-MCP console script.

## Additional Confirmed Risks

- Sync reconstruction forces external/`kernle:sync` identity while omitting parts of lineage/privacy, then writes through replacement paths (`kernle/storage/sync_engine.py:1234-1385`). Remote sync should remain deferred until policy and isolation are fixed.
- Legacy emotion mixins and the default Stack emotion component can both infer on one CLI episode save, doubling provider calls and potentially overwriting explicit fields (`kernle/features/emotions.py:305-357`; `kernle/stack/sqlite_stack.py:482-519`). This is direct evidence that API/component consolidation is needed.
- Model processing applies JSON parsing but lacks a single transition schema and bounded, source-ID-checked validation before persistence (`kernle/processing.py:1137-1152,1288-1494`). Qualification cases must include malformed and adversarial outputs.
- MCP's deep schema validation depends on optional `jsonschema`; its fallback validates only a subset of constraints (`kernle/mcp/server.py:37-49,169-230`). Plugin tool inputs therefore need a declared validator or equivalent complete local validation.
- Checkpoint/memory file writes are not consistently locked, atomic, crash-safe, or permission-restricted (`kernle/core/checkpoint.py:48-76`; `kernle/entity.py:1360-1383`). Address after the release-blocking data-path defects with one secure atomic-write helper and concurrency tests.

## Verification Performed

| Check | Result |
|---|---|
| Clean Python 3.12 isolated editable install with `.[dev]` | Passed |
| Python collection | 6,797 tests collected |
| Full Python suite | 6,787 passed, 10 skipped in 190.08s |
| Python branch coverage | 89.31%; 85% gate passed |
| Direct global `pytest tests/` without install | Failed collection: eight MCP modules could not import `mcp` |
| Ruff | Passed |
| Black check | Failed: `tests/test_mcp.py` would be reformatted |
| Python compileall | Passed |
| OpenClaw Vitest | 54 passed across six files |
| OpenClaw TypeScript build | Passed |
| OpenClaw package dry run | Passed in the current worktree; clean-checkout build/package ordering remains untested |
| OpenClaw lint | Failed: `eslint` script exists but no dependency/config is installed |
| Python sdist/wheel build and Twine validation | Passed |
| Base wheel `kernle-mcp --help` smoke | Failed: console script is always installed while optional `mcp` dependency is absent |
| Last remote `Test` workflow | Passed on `origin/main` commit `6a990c3`, 2026-02-20; it does not cover 23 local commits |

The 10 Python skips are conditional, including sqlite-vec cases and one inaccessible Anxiety component case. Conditional skips are not necessarily defects, but a fleet must declare whether they are expected in its target environment.

The full-suite and coverage counts are valid, but those runs were not data-isolated because existing fixtures use the wrong environment variable. Do not repeat them until Bundle 00's isolation guard is in place.

## Test Material Sufficiency

| Proposed next step | Sufficient now? | What is missing |
|---|---:|---|
| Authority/docs inventory and cleanup | Yes | Owner must choose the intended `audits/`/`dev-docs/` outcome |
| Reproducible Python baseline and docs smoke | Partial | No owner input needed; first fix test data isolation, then checkout commands/examples |
| OpenClaw failed-capture/stack-routing repair | Yes | Add failure and per-session routing tests; real-host canary still needed before support claim |
| Atomic update outcome repair | Yes | Add failing return/hook tests first |
| Migration failure handling | Mostly | Logic can be fixed; supported-release snapshots are needed to claim upgrade safety |
| Multi-stack isolation repair | Partial | Owner must choose one database per stack (recommended) or stack-scoped composite keys; add collision/queue fixtures |
| Select canonical API and write model | No | Owner decision; current tests protect both architectures |
| Privacy and consent enforcement | No | Approved access truth table and requester semantics |
| Provenance integrity | No | Approved immutability/repair matrix and trusted-actor rule |
| Semantic memory quality | No | Human-labelled realistic multi-session corpus and scoring rubric |
| MCP/host compatibility claim | No | Versioned real payload fixtures and installed-product canary |
| Performance claim | No | Workload definitions, generated corpora, and thresholds |
| Hosted cloud/dashboard | No | Service repo/environment and design authority are absent; recommended deferral |

## Why the Existing Suite Is Not Enough

The suite is valuable regression material: 201 Python test files, 6,797 cases, and 333 contract cases are broad, fast, and green in a clean environment. But the contract registry tests only SQLite and disables provenance in Stack fixtures; 116 files use `unittest.mock`; no unit/integration/E2E/live taxonomy is enforced. The golden corpus is only eight small files (188 lines, about 5.8 KB), and its expected model behavior comes from deterministic prompt-keyed mocks. That verifies plumbing and counts, not whether Kernle remembers the right facts, rejects unsafe interpretations, preserves privacy across actors, or remains useful over realistic sessions. High line/branch coverage cannot replace those product or security oracles.

## Recommended Next Steps

1. Stop full-suite execution; fix `KERNLE_DATA_DIR`/temporary-HOME isolation and prepare an owner-reviewed cleanup inventory for the real local database.
2. Confirm the `dev-docs/` move and approve/replace the narrow owner-only alpha plus decisions 0001-0006.
3. Repair mechanical authority/bootstrap and migration atomicity; do only retained-type atomic-write work.
4. Add three minimal workflow cases plus owner-only privacy and provenance truth tables as red contracts.
5. Consolidate to the one retained API before threading policy.
6. Implement the pure owner-only request/provenance policy, per-stack database isolation, and canonical SDK boundary serially.
7. Fan out only two disjoint lanes: MCP and one export/restore path.
8. Run installed-wheel/docs/MCP release qualification. OpenClaw, sync, providers, extra memory types, and semantic-quality work remain separate post-alpha claims.

## Fleet Readiness Gate

Do not fan out feature work until all are true:

1. Root authority and the internal-docs location are clean, tracked, and accurate.
2. One alpha contract names the primary actor, workflows, non-goals, canonical API, write/promotion model, supported host, and release meaning.
3. Every public surface is labelled implemented, partial, planned, deferred, or removed.
4. Approved workflow, privacy, provenance, and migration fixtures exist.
5. A fresh environment runs targeted and full tests using documented commands.
6. Public quickstart and one selected host integration pass installed-package smoke tests.
7. Privacy and provenance failures are encoded as red contract tests before implementation.
8. An independent auditor finds no unresolved P0/P1 issue in the claimed slice.

## Live Backlog Assessment

The public repository currently has 101 open issues and no open pull requests. Milestones contain 36 Cleanup, 26 Pondering, 13 Skills, 14 Sync, 8 Privacy, and 4 Aspirational issues. Most activity predates the May internal roadmap. That sequence puts privacy after new cognitive features, contrary to the dependency order required by the current architecture. Archive or re-triage it after the product contract; do not hand the raw backlog to agents as implementation authority.
