# Agent Improvement Lab

A fictional customer-support agent that recommends resolutions, evaluates its own traces, and tests versioned procedural updates. Gemini chooses read-only information tools inside a bounded LangGraph loop. A separate controller can propose a skill patch after an externally scored failure. No order, payment, refund, shipment, or warranty action is executed.

## Setup with uv

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/getting-started/installation/), and a Gemini API key.

```bash
uv sync --locked
cp .env.example .env
# Put GEMINI_API_KEY in .env, then run:
uv run --locked python demo.py case --case-id C-14
```

`GEMINI_MODEL` defaults to `gemini-2.5-flash`. To send traces to LangSmith, set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT=agent-improvement-lab` in `.env`. The same local JSON trajectory is saved when LangSmith is unavailable. Keep `.env` private.

For the customer chat and operator viewer:

```bash
uv sync --locked --extra dashboard
uv run --locked --extra dashboard streamlit run app/customer_chat.py
uv run --locked --extra dashboard streamlit run app/demo_dashboard.py
```

The customer chat is a small demonstration using fictional orders. Browser session state holds the conversation; restarting Streamlit starts a new conversation. Case memory is separately persisted in SQLite. There is no ownership authentication.

## Six stages

```bash
uv run --locked python demo.py baseline
uv run --locked python demo.py diagnose
uv run --locked python demo.py propose
uv run --locked python demo.py validate
uv run --locked python demo.py promote
uv run --locked python demo.py compare
uv run --locked python demo.py publish-langsmith
uv run --locked python demo.py safeguard
uv run --locked python demo.py rollback
```

`demo.py demo` runs the sequence, including the bad-update probe. `--run-dir PATH` selects a run; otherwise commands use `runs/latest.txt`. `--trace-id UUID` selects a specific failed training trace. `case --case-id C-07 --version active` checks a promoted skill. `--skill-version ID` loads a particular stored version, and `--memory-snapshot PATH` loads an existing case-memory SQLite snapshot for a single case. Paired validation and holdout runs create their own frozen snapshots. `reject` records an operator rejection. `seed-memory` inserts a clearly labeled fictional prior day-eight defect recommendation for the memory walkthrough. `record-memory --case-id C-03` records a proposed summary after a saved single-case assessment. `publish-langsmith` publishes the saved heldout outputs as LangSmith experiments without making new Gemini calls.

The baseline run scores eight training cases and queues one job for each observed failure. The controller diagnoses one selected job using trace step IDs, proposes a skill patch against `skills.md`, reviews it against current policies, runs paired validation and an independent safety suite, then promotes only if at least one validation failure is fixed with no passing-case regression or new critical failure. The promoted text is saved as `runs/<run>/active-skills.md` and in SQLite. `rollback` restores its previous active version across process restarts. The eight heldout cases are run once for final comparison; subsequent `compare` calls read the saved report.

`skills.md` is the initial v1 procedure. It deliberately omits an explicit final-sale defect precedence rule. A real Gemini run may or may not fail on that gap; diagnose only an observed failure. Saved deterministic tests are orchestration checks, not live Gemini findings. The intentionally misleading refund shortcut is tested as an isolated candidate and remains inactive. The safety layer also checks current order and policy facts before any non-escalation recommendation is shown.

The three functions in `agent/tools.py` are `get_order_details`, `search_policy`, and `create_resolution`. The first two read fictional data; the third validates a recommendation. `retrieve_case_memory` is an optional read from `agent/case_memory.py`. A controller-gated write records only short, scoped summaries with provenance and a proposed or human-confirmed status. A proposed action is never recorded as an executed outcome. Tombstones preserve corrections in the audit trail.

`runs/` contains per-case trajectories, evaluator reports, improvement jobs with skill diffs and gate decisions, versioned SQLite data, and a final comparison. The reports include trace IDs and LangSmith run IDs when available. Inspect actual scores before a presentation; no improvement is guaranteed by a model call.

## Tests

```bash
uv run --locked python -m unittest discover -s tests -v
```

The tests use deterministic fake model decisions to verify graph bounds, optional tool trajectories, memory scope, chat intake, the policy gate, heldout reuse, and rollback. They do not establish a live Gemini success rate.
