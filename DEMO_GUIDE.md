# Presenter guide · 20 minutes

## Prepare a real fallback

Configure Gemini and LangSmith, then run `uv run --locked python demo.py demo` once before presenting. Keep the generated `runs/<timestamp>/` folder. Open the operator viewer with `uv run --locked --extra dashboard streamlit run app/demo_dashboard.py`. Inspect `baseline-train.json`, the saved job, `comparison.json` if promotion passed, and each local trace. Copy the actual LangSmith run URL from LangSmith for the case you will show. The test suite uses a fake model and must not be described as a live Gemini result. If Gemini produced no training failure or the candidate did not pass the gate, say so and show the saved evidence instead of promising a win.

The current local fallback is `runs/real-final-20261010/` (ignored by Git). That real Gemini baseline passed 6/8 training cases. Its paired validation passed 8/8 for both versions, so the candidate was rejected for no measured improvement. A seeded memory note was retrieved on a later C-03 request, and Gemini recommended warranty. The bad-update probe was rejected by policy review. LangSmith ingestion returned HTTP 403 with the configured key, so this fallback currently has local trajectories rather than working remote trace links. Re-run or copy the ignored folder when presenting on another machine.

## 0–2 min · customer chat

Launch `uv run --locked --extra dashboard streamlit run app/customer_chat.py`. Ask about a fictional order without its ID, then supply `O-103`. Show the polite intake question and later recommendation. The order-ID intake is a deterministic guard; the subsequent LangGraph `decide → dispatch → decide` path chooses information tools. The customer view contains no internal skill or trace data.

## 2–6 min · observed skill gap

Open `skills.md` v1 and a *failed* trace under `traces/baseline-train/`. C-14 is a useful final-sale defect case only if Gemini actually mishandled it. Point to the skill hash, selected policy ID, order facts, decision output, final safety intervention, and evaluator result. If C-14 passed, use another actual failure and classify that evidence instead.

## 6–9 min · diagnose

Run `uv run --locked python demo.py diagnose --run-dir runs/<timestamp>` or open the saved `jobs/<trace-id>.json`. The category should be supported by trace step IDs: wrong or missing tool selection, a missing skill instruction, stale memory use, or answer synthesis. Show the current policy text supporting the diagnosis.

## 9–12 min · propose and validate

Run `propose`, then `validate` with the same `--run-dir`. Show the unified skill diff, source trace ID, policy review, paired baseline/candidate validation reports, and separate safety cases. A candidate must fix an observed validation failure without regressing a passing case or causing a critical policy failure. `promote` records the gate result; a rejection is still inspectable. The controller changes skill text, never authoritative policies or evaluator labels.

## 12–16 min · compare and memory

If promoted, run `compare` once and inspect the per-case heldout reports. Repeating the command reads the saved comparison. Run `seed-memory` for the clearly labeled fictional day-eight case, then `uv run --locked python demo.py case --case-id C-03 --run-dir runs/<timestamp>` and inspect the later day-75 request for the same demo customer in the operator trace. The previous refund recommendation is context only; current P-DEFECT and order facts require warranty. `agent/case_memory.py` marks a different order or policy version stale and keeps proposed outcomes separate from human-confirmed ones. Do not describe the seed as a real customer record.

## 16–18 min · safety and rollback

Run `safeguard` to inspect the isolated bad refund shortcut. Show the policy review, safety case scores, rejected version, and unchanged active pointer. If a good version was promoted earlier, `rollback` restores its parent version. No business action occurs in either run.

## 18–20 min · limits

The data and APIs are fictional. Gemini recommendations may vary; the policy guard can escalate unsafe ones. The local evaluator covers the five fixture policies, not every real-world support nuance. Memory is scoped and audit-friendly, while Streamlit conversation state lasts only for that browser session. This demo changes a procedural skill, not model weights. Human review remains appropriate for business policy changes, which this controller does not permit.
