# Agent Improvement Lab

A six-stage, 20-minute demo of a self-improving customer-support resolution agent. Gemini makes the policy selection and recommendation; fictional order and policy APIs are mocked. LangGraph runs the agent, SQLite versions procedural lessons, and LangSmith can trace runs and compare evaluation experiments. No tool issues a real refund or changes an order.

## Repository

```text
agent/
  graph.py          LangGraph and Gemini model
  tools.py          get_order_details, search_policy, create_resolution
  prompts.py        structured outputs and baseline instructions
  memory.py         LangGraph store cache of active procedural memory
  data/             fictional orders and authoritative policies
learning/
  reflection.py     lesson proposal and policy review
  lesson_store.py   SQLite version history and rollback
  promotion_gate.py validation gate
evals/
  train_cases.json
  validation_cases.json
  heldout_cases.json
  evaluators.py     deterministic scores and local reports
  langsmith_experiments.py  remote dataset and experiment comparison
experiments/
  baseline.json     baseline configuration
  candidate.json    candidate artifact specification
app/
  demo_dashboard.py six-stage artifact viewer
demo.py             presentation commands
DEMO_GUIDE.md       presenter notes
```

## Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.11+, and a Gemini API key. `uv sync` creates and manages `.venv` automatically.

```bash
uv sync --locked
cp .env.example .env
# Add GEMINI_API_KEY to .env. Keep .env private.
uv run python demo.py demo
UV_HTTP_TIMEOUT=120 uv sync --locked --extra dashboard
uv run --locked --extra dashboard streamlit run app/demo_dashboard.py
```

`uv.lock` pins the resolved dependencies. The dashboard is optional and downloads large packages, so install it only when you want the Streamlit viewer.

To use LangSmith, set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING=true`, and `LANGSMITH_PROJECT=agent-improvement-lab` in `.env`. The graph and Gemini calls then appear in LangSmith. `uv run python demo.py compare --publish-langsmith` also creates a heldout dataset and runs fresh baseline and learned experiments with deterministic evaluators. That command makes additional Gemini calls. Local JSON traces and scores work without a LangSmith key.

The default model is `gemini-2.5-flash`; set `GEMINI_MODEL` to change it. The full 24-case demo makes many model calls and may incur Gemini and LangSmith usage charges. Run it once before presenting, then use the saved artifacts and one fresh case live.

## Six stages

```bash
uv run python demo.py case --case-id C-01              # 1–2: agent and visible failure
uv run python demo.py baseline                         # 3: 8 training cases
uv run python demo.py learn                            # 4: reflect, policy-check, validate, promote
uv run python demo.py compare                          # 5: 8 untouched holdout cases, both versions
uv run python demo.py case --case-id C-09 --version active  # fresh unseen-case check
uv run python demo.py safeguard                        # 6: reject the bad refund rule
uv run python demo.py rollback                         # restore the prior active version if needed
```

`uv run python demo.py demo` runs all six stages in one command. Artifacts go to `runs/<timestamp>/`: traces, score reports, `candidate.json`, `comparison.json`, `bad-update.json`, and `lessons.sqlite`. `runs/latest.txt` selects the last run for subsequent commands. The generated lesson is an instruction version in SQLite; no Gemini model weights are changed.

The lesson sees failed **training** traces, evaluator feedback, and policy text. A separate policy review and validation suite gate promotion. The holdout set is never included in reflection or validation. If there is no training failure or no validated improvement, the result says so instead of inventing a win. The intentional bad rule is rejected for contradicting the defect policy, and its safety-suite failures remain inspectable. `rollback` is available for a previously promoted lesson.

The deterministic scores cover resolution correctness, required tool usage, policy citation compliance, and a narrow unsupported-claim check for falsely completed actions. They do not replace human review of real support language.

## Tests

```bash
uv run python -m unittest discover -s tests -v
```

Tests use a fake model to verify orchestration, gating, and rollback. Their scores are simulations, never presented as live Gemini results. A real Gemini run requires your API key.

## References

- [LangGraph graph API](https://docs.langchain.com/oss/python/langgraph/use-graph-api)
- [LangSmith evaluation](https://docs.langchain.com/langsmith/evaluation-types)
- [Google Gen AI Python SDK](https://googleapis.github.io/python-genai/)
