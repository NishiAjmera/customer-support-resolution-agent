"""Mock business tools; all are read-only or recommend-only."""

import json
from pathlib import Path
from typing import Any

from .prompts import Resolution


DATA_DIR = Path(__file__).parent / "data"


def load_records(filename: str) -> dict[str, dict]:
    return {row["id"]: row for row in json.loads((DATA_DIR / filename).read_text())}


def get_order_details(order_id: str, orders: dict[str, dict]) -> dict:
    """Return fictional order facts for an existing order ID."""
    if order_id not in orders:
        raise ValueError(f"Unknown order: {order_id}")
    return orders[order_id].copy()


def search_policy(policy_ids: list[str], policies: dict[str, dict]) -> list[dict]:
    """Fetch exact policy text selected from the public policy catalog."""
    if not policy_ids or any(pid not in policies for pid in policy_ids):
        raise ValueError("The agent requested an unknown or empty policy selection")
    return [policies[pid].copy() for pid in dict.fromkeys(policy_ids)]


def create_resolution(model: Any, prompt: str) -> dict:
    """Generate a recommendation; this never executes a business action."""
    return model.generate(prompt, Resolution).model_dump()
