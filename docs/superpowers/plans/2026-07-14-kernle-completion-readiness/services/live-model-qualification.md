# Service: Live Model Qualification

## Purpose

Run a small, explicit canary against one approved provider after deterministic qualification passes. Live output is evidence about integration and quality distributions, not a replacement for contract tests.

## Required Config and Secrets

Track names only:

- `KERNLE_TEST_LIVE_PROVIDER`
- `KERNLE_TEST_LIVE_MODEL`
- `KERNLE_TEST_MAX_USD_PER_RUN`
- `KERNLE_TEST_DAILY_MAX_USD`
- `KERNLE_TEST_TIMEOUT_SECONDS`
- exactly one of `CLAUDE_API_KEY`/`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`

These opt-in names are proposed and must be wired/documented by the proof bundle. Never commit values.

## Human Setup Steps

1. Select one provider/model and approve cost/time limits.
2. Put values in `~/.config/kernle/fleet.env` with user-only permissions, or another explicitly approved secret store.
3. Confirm fixtures are sanitized and the provider's data handling is acceptable.
4. Run a single smoke before a batch.

Permission and load check:

```bash
umask 077
mkdir -p ~/.config/kernle
chmod 700 ~/.config/kernle
chmod 600 ~/.config/kernle/fleet.env
set -a
source ~/.config/kernle/fleet.env
set +a
python -c 'import os; p=os.environ.get("KERNLE_TEST_LIVE_PROVIDER"); assert p; assert os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"); print(f"live provider configured: {p}")'
```

The command reports only the provider name, never a key. The fleet must add one read-only, one-request canary at `tests/live/test_selected_provider.py`; the intended command is `python -m pytest tests/live/test_selected_provider.py -m live_provider -q`. That test does not exist today, so live access remains blocked rather than guessed.

## Allowed and Forbidden Behavior

- Opt-in only; deterministic CI must not call a provider.
- No private/raw production memories.
- No automatic retries beyond the approved bound.
- No model change or fallback that hides a failed selected-provider test.
- Record provider/model/version, prompt/template hash, cost, latency, and outcome without secrets.

## Test and Smoke Requirements

- Recorded/redacted success, malformed response, timeout, rate limit, and provider error shapes.
- Human-labelled qualification rubric and abstention/unsafe-output checks.
- Repeat enough cases to report distribution and variance; do not assert exact prose.

## Current Blockers

- No owner-approved provider/model/budget.
- No human-labelled multi-session corpus.
- OpenAI packaging advertises a nonexistent extra.
- Existing golden tests use deterministic mock responses only.
