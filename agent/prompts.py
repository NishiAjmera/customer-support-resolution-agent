"""Typed model outputs for recommendations and skill review."""
from typing import Literal

from pydantic import BaseModel, Field


class Resolution(BaseModel):
    action: Literal["refund", "replacement", "warranty", "deny", "escalate"]
    policy_ids: list[str] = Field(default_factory=list)
    reason: str
    customer_reply: str


class LessonProposal(BaseModel):
    rules: list[str] = Field(min_length=1, max_length=4)


class PolicyReview(BaseModel):
    supported: bool
    explanation: str
