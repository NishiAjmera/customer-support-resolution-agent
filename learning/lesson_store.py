"""SQLite-backed, versioned procedural memory."""

from __future__ import annotations

import json
import sqlite3
import hashlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from agent.skills import SKILL_FILE


class LessonStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rules_json TEXT NOT NULL,
                    source_cases_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parent_id INTEGER,
                    policy_review_json TEXT,
                    validation_json TEXT,
                    rejection_reason TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(versions)")}
            for name, sql_type in (("content", "TEXT"), ("content_hash", "TEXT"),
                                   ("diagnosis_json", "TEXT"), ("evidence_json", "TEXT")):
                if name not in columns:
                    db.execute(f"ALTER TABLE versions ADD COLUMN {name} {sql_type}")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    @staticmethod
    def _decode(row):
        if row is None:
            return None
        result = dict(row)
        for key in ("rules_json", "source_cases_json", "policy_review_json", "validation_json",
                    "diagnosis_json", "evidence_json"):
            value = result.pop(key)
            result[key.removesuffix("_json")] = json.loads(value) if value else None
        return result

    def create_candidate(self, rules: list[str], source_cases: list[str], content: str | None = None,
                         diagnosis: dict | None = None, evidence: list[str] | None = None) -> dict:
        if not rules:
            raise ValueError("A lesson candidate must have at least one rule")
        current = self.active()
        old = current["content"] if current and current.get("content") else SKILL_FILE.read_text()
        content = content or (old + "\n## Validated lessons\n" + "\n".join(f"- {rule}" for rule in rules) + "\n")
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        with self._connect() as db:
            cursor = db.execute(
                """INSERT INTO versions (rules_json, source_cases_json, status, created_at,
                   content, content_hash, diagnosis_json, evidence_json, parent_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (json.dumps(rules), json.dumps(source_cases), "draft", datetime.now(timezone.utc).isoformat(),
                 content, content_hash, json.dumps(diagnosis) if diagnosis else None,
                 json.dumps(evidence or []), current["id"] if current else None),
            )
            version_id = cursor.lastrowid
        return self.get(version_id)

    def get(self, version_id: int) -> dict | None:
        with self._connect() as db:
            return self._decode(db.execute("SELECT * FROM versions WHERE id = ?", (version_id,)).fetchone())

    def active(self) -> dict | None:
        with self._connect() as db:
            return self._decode(db.execute("SELECT * FROM versions WHERE status = 'active' ORDER BY id DESC LIMIT 1").fetchone())

    def list_versions(self) -> list[dict]:
        with self._connect() as db:
            return [self._decode(row) for row in db.execute("SELECT * FROM versions ORDER BY id DESC")]

    def record_review(self, version_id: int, review: dict) -> None:
        with self._connect() as db:
            db.execute("UPDATE versions SET policy_review_json = ? WHERE id = ? AND status = 'draft'",
                       (json.dumps(review), version_id))

    def record_validation(self, version_id: int, validation: dict) -> None:
        with self._connect() as db:
            db.execute("UPDATE versions SET validation_json = ? WHERE id = ? AND status = 'draft'",
                       (json.dumps(validation), version_id))

    def reject(self, version_id: int, reason: str) -> dict:
        with self._connect() as db:
            db.execute("UPDATE versions SET status = 'rejected', rejection_reason = ? WHERE id = ? AND status = 'draft'",
                       (reason, version_id))
        return self.get(version_id)

    def promote(self, version_id: int) -> dict:
        candidate = self.get(version_id)
        if not candidate or candidate["status"] != "draft":
            raise ValueError("Only a draft lesson can be promoted")
        if not candidate["policy_review"] or not candidate["policy_review"].get("supported"):
            raise ValueError("Policy review must pass before promotion")
        if not candidate["validation"] or not candidate["validation"].get("accepted"):
            raise ValueError("Validation gate must pass before promotion")
        current = self.active()
        with self._connect() as db:
            db.execute("UPDATE versions SET status = 'archived' WHERE status = 'active'")
            db.execute("UPDATE versions SET status = 'active', parent_id = ? WHERE id = ?",
                       (current["id"] if current else None, version_id))
        self._write_active_artifact(candidate["content"])
        return self.get(version_id)

    def rollback(self) -> dict | None:
        current = self.active()
        if current is None:
            return None
        with self._connect() as db:
            db.execute("UPDATE versions SET status = 'rolled_back' WHERE id = ?", (current["id"],))
            if current["parent_id"] is not None:
                db.execute("UPDATE versions SET status = 'active' WHERE id = ?", (current["parent_id"],))
        active = self.active()
        self._write_active_artifact(active["content"] if active and active.get("content") else SKILL_FILE.read_text())
        return self.active()

    def _write_active_artifact(self, content: str) -> None:
        artifact = self.path.parent / "active-skills.md"
        temporary = artifact.with_suffix(".tmp")
        temporary.write_text(content)
        temporary.replace(artifact)
