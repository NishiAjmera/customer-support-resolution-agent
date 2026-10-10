"""Explicit selection and versioned loading of the support procedure."""

from __future__ import annotations

import hashlib
from pathlib import Path

SKILL_FILE = Path(__file__).resolve().parents[1] / "skills.md"


def digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def select_skill(message: str, store=None, version_id: int | None = None) -> dict:
    """This demo has one support-domain skill; selection is still observable."""
    version = store.get(version_id) if store and version_id else (store.active() if store else None)
    if version_id and not version:
        raise ValueError(f"Unknown skill version {version_id}")
    content = version["content"] if version and version.get("content") else SKILL_FILE.read_text()
    return {"id": "support-resolution", "version": version["id"] if version else "v1",
            "hash": digest(content), "content": content}


def patched_skill(old: str, rule: str) -> str:
    marker = "\n## Evidence and trust"
    if marker not in old:
        raise ValueError("Skill lacks the expected evidence section")
    return old.replace(marker, f"\n- {rule}\n{marker}", 1)
