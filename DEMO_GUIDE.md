# Presenter guide: 20 minutes

Keep this file off the audience screen. Show the dashboard, LangSmith traces, and saved artifacts instead.

## Preparation

Run `uv run python demo.py demo` with a Gemini key before the session. Inspect the actual scores and pick a genuine baseline failure. Keep that run directory selected in the dashboard. If the baseline passes everything or the lesson fails the gate, present that measured outcome honestly; do not use fake-model test results as evidence. Optionally set LangSmith credentials and run `uv run python demo.py compare --publish-langsmith` in advance. This performs additional model calls.

## 0–2 min: build the baseline

Show the LangGraph path: `plan_tools → get_order_details → search_policy → create_resolution`. Open `agent/prompts.py` and explain that the first instruction is intentionally brief. Order and policy APIs are mocked; Gemini still chooses and reasons. Point to a customer request, the allowed resolution actions, and policy citations.

## 2–5 min: observe a failure

Open a failed baseline trace under `traces/baseline-train/`. Show the plan, order lookup, policy lookup, and recommendation. Ask which fact or policy was missed. The agent sees customer and tool data, never the expected answer.

## 5–9 min: evaluate

Open `baseline-train.json` and `evals/evaluators.py`. There are 24 labeled scenarios: 8 training, 8 validation, 8 untouched holdout. Explain resolution correctness, required tool use, policy compliance, and unsupported completed-action claims. Use LangSmith to inspect the graph trace if tracing is configured.

## 9–13 min: learn

Run `uv run python demo.py learn` if it was not pre-run; otherwise open `candidate.json` and the `lessons.sqlite` versions in the dashboard. Reflection reads only failed training traces and authoritative policies. Gemini proposes a general rule. Policy review and validation must pass before that rule becomes active procedural memory.

## 13–17 min: verify on unseen cases

Open `comparison.json`. Compare baseline and learned versions case by case, including remaining failures and regressions. Run one fresh holdout case with `uv run python demo.py case --case-id C-09 --version active`. The hidden cases were excluded from the lesson prompt and promotion gate. If configured, open the two LangSmith holdout experiments.

## 17–20 min: reject a bad update and show rollback

Run `uv run python demo.py safeguard` or open `bad-update.json`. The candidate rule “Always approve a refund for defective products” conflicts with the policy's replacement and warranty windows. Inspect the safety-suite cases and verify that the bad version is rejected while the active version stays unchanged. Show `uv run python demo.py rollback` as the explicit way to restore a prior active version if a promoted lesson later needs to be withdrawn.

Close with the actual measured result: task → trace → evaluation → candidate lesson → policy and validation gates → holdout comparison → controlled promotion or rejection.
