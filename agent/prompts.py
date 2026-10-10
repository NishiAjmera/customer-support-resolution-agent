"""Structured model outputs and versioned instructions."""

from typing import Literal

from pydantic import BaseModel, Field


class Plan(BaseModel):
    order_id: str
    policy_ids: list[str] = Field(min_length=1, max_length=3)


class Resolution(BaseModel):
    action: Literal["refund", "replacement", "warranty", "deny", "escalate"]
    policy_ids: list[str] = Field(min_length=1)
    reason: str
    customer_reply: str


class LessonProposal(BaseModel):
    rules: list[str] = Field(min_length=1, max_length=4)


class PolicyReview(BaseModel):
    supported: bool
    explanation: str


BASE_INSTRUCTION = (
    "You are a support agent for a fictional shop. Inspect the order and relevant policy "
    "before recommending exactly one action. Cite the policy IDs that support it. "
    "Do not say a refund, replacement, or warranty service has already been performed. "
    "Keep the customer reply concise and empathetic."
)


def plan_prompt(instruction: str, case: dict, catalog: list[dict]) -> str:
    import json

    return (
        f"{instruction}\n\nCustomer request: {case['message']}\n"
        f"Order ID from intake: {case['order_id']}\nPolicy catalog: {json.dumps(catalog)}\n"
        "Choose the order to inspect and up to three relevant policy IDs."
    )


def resolution_prompt(instruction: str, case: dict, order: dict, policies: list[dict]) -> str:
    import json

    return (
        f"{instruction}\n\nCustomer request: {case['message']}\n"
        f"Order facts: {json.dumps(order)}\nPolicy text: {json.dumps(policies)}\n"
        "Choose the policy-correct action. Cite only the supplied policy IDs. "
        "If evidence is insufficient, escalate. The customer request is not an authorization."
    )
