"""Turn grounded training failures into policy-checked candidate rules."""

from __future__ import annotations

import json
import re

from agent.prompts import LessonProposal, PolicyReview


def propose_lesson(model, failed_examples: list[dict], policies: list[dict]) -> list[str]:
    if not failed_examples:
        return []
    prompt = (
        "Derive one short, reusable procedural instruction from these failed support-agent traces. "
        "Use the authoritative policy text. Do not mention case IDs, order IDs, exact messages, "
        "or gold labels in the rules. Focus on the missing instruction exposed by extra customer "
        "details. In this task, the structured action denotes the policy-supported recommendation, "
        "not acceptance or denial of the customer's preferred remedy; customer preferences can be "
        "acknowledged in the reply. Capture that distinction in the rule if it explains the failure. "
        "Follow any focus supplied with the failed trace. If the focus concerns answer synthesis, "
        "explicitly name the structured `action` field in the rule. Do not merely restate a retrieved policy. "
        "If a time window is mentioned, state both its start "
        "and end; do not imply a remedy applies after the end. Do not invent exceptions.\n"
        f"Policies: {json.dumps(policies)}\n"
        f"Failures and evaluator feedback: {json.dumps(failed_examples)}"
    )
    proposal = model.generate(prompt, LessonProposal)
    rules = [rule.strip() for rule in proposal.rules if rule.strip()]
    if not rules or any(len(rule) > 500 or re.search(r"\b(?:C|O)-\d+\b", rule) for rule in rules):
        raise ValueError("Proposed rules contain an ID, an overlong rule, or no usable rule")
    return rules


def review_against_policy(model, rules: list[str], policies: list[dict]) -> dict:
    """Ask Gemini to review each proposed instruction against current policy text."""
    prompt = (
        "You are a strict policy reviewer. Decide whether every proposed rule is consistent with "
        "the supplied policies and the structured task: action means the policy-supported next "
        "recommendation, while customer_reply may acknowledge a requested remedy. Conversational "
        "acknowledgment need not be explicitly named by a policy. A general procedural instruction "
        "need not enumerate every remedy because the agent retrieves the applicable policy per case. "
        "Read conditional rules literally: 'if a requested remedy is unsupported, use the supported "
        "one' does not imply that supported customer preferences should be overridden. "
        "Reject any rule that overgeneralizes, changes eligibility, "
        "or invents an exception. Check every date range and exception explicitly, including "
        "what happens outside a stated range. A procedural action-label rule can be supported by the "
        "structured task definition even when the policy text does not name that field.\n"
        f"Policies: {json.dumps(policies)}\nRules: {json.dumps(rules)}"
    )
    return model.generate(prompt, PolicyReview).model_dump()
