"""Scoped fictional case memory; recommendations are never confirmed outcomes."""
from __future__ import annotations

import json
import sqlite3
import uuid
import hashlib
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langsmith.run_helpers import trace


class CaseMemory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS cases (
                memory_id TEXT PRIMARY KEY, customer_id TEXT NOT NULL, order_id TEXT NOT NULL,
                issue_type TEXT NOT NULL, issue_summary TEXT NOT NULL, policy_ids_json TEXT NOT NULL,
                policy_hash TEXT NOT NULL, recommendation TEXT, outcome_status TEXT NOT NULL,
                source_type TEXT NOT NULL, source_trace_id TEXT NOT NULL, created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL, supersedes_id TEXT, tombstoned INTEGER NOT NULL DEFAULT 0,
                correction_reason TEXT)""")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        finally:
            db.close()

    def write_case_memory(self, *, customer_id: str, order_id: str, issue_type: str,
                          issue_summary: str, policy_ids: list[str], policy_hash: str,
                          recommendation: str | None, source_trace_id: str,
                          source_type: str = "agent_recommendation",
                          outcome_status: str = "proposed", human_confirmation: bool = False,
                          supersedes_id: str | None = None) -> str:
        if outcome_status == "human_confirmed" and not human_confirmation:
            raise ValueError("Verified outcome requires explicit human confirmation")
        if outcome_status not in {"proposed", "human_confirmed", "unknown"}:
            raise ValueError("Invalid outcome status")
        if source_type not in {"agent_recommendation", "tool_verified_fact", "human_confirmation", "fixture"}:
            raise ValueError("Invalid source type")
        if len(issue_summary) > 240 or "@" in issue_summary:
            raise ValueError("Store only a short fictional case summary")
        memory_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with trace("memory.write", run_type="tool", inputs={"customer_id": customer_id,
                   "order_id": order_id, "source_trace_id": source_trace_id}) as span:
            with self._db() as db:
                db.execute("""INSERT INTO cases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, NULL)""",
                           (memory_id, customer_id, order_id, issue_type, issue_summary,
                            json.dumps(policy_ids), policy_hash, recommendation, outcome_status,
                            source_type, source_trace_id, now.isoformat(),
                            (now + timedelta(days=180)).isoformat(), supersedes_id))
                db.commit()
            span.end(outputs={"memory_id": memory_id, "status": outcome_status})
        return memory_id

    def retrieve(self, customer_id: str | None, order_id: str | None, issue: str,
                 current_policy_hash: str, limit: int = 3) -> list[dict]:
        if not customer_id or not order_id:
            return []
        issue_type = self.issue_type(issue)
        now = datetime.now(timezone.utc).isoformat()
        with self._db() as db:
            rows = db.execute("""SELECT * FROM cases WHERE customer_id = ? AND issue_type = ?
                AND tombstoned = 0 AND expires_at > ? ORDER BY (order_id = ?) DESC, created_at DESC LIMIT ?""",
                (customer_id, issue_type, now, order_id, limit)).fetchall()
        result = []
        for row in rows:
            value = dict(row)
            value["policy_ids"] = json.loads(value.pop("policy_ids_json"))
            value["stale"] = value["order_id"] != order_id or value["policy_hash"] != current_policy_hash
            value["trust_level"] = "verified" if value["outcome_status"] == "human_confirmed" else "context_only"
            result.append(value)
        return result

    @staticmethod
    def issue_type(issue: str) -> str:
        issue_lower = issue.lower()
        return ("defect" if any(word in issue_lower for word in ("defect", "broken", "failed", "fault",
                                                                  "stopped working"))
                else "wrong_item" if "wrong" in issue_lower else "delivery" if "deliver" in issue_lower
                else "return")

    def has_relevant(self, customer_id: str | None, issue: str) -> bool:
        if not customer_id:
            return False
        with self._db() as db:
            row = db.execute("""SELECT 1 FROM cases WHERE customer_id = ? AND issue_type = ?
                AND tombstoned = 0 AND expires_at > ? LIMIT 1""",
                (customer_id, self.issue_type(issue), datetime.now(timezone.utc).isoformat())).fetchone()
        return row is not None

    def snapshot(self, destination: Path) -> tuple["CaseMemory", str]:
        """Copy a consistent SQLite snapshot for paired, read-only experiments."""
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            destination.unlink()
        source = sqlite3.connect(self.path)
        target = sqlite3.connect(destination)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        return CaseMemory(destination), hashlib.sha256(destination.read_bytes()).hexdigest()[:16]

    def tombstone(self, memory_id: str, reason: str, source_trace_id: str) -> None:
        if not reason or not source_trace_id:
            raise ValueError("Correction provenance is required")
        with self._db() as db:
            cursor = db.execute("UPDATE cases SET tombstoned = 1, correction_reason = ? WHERE memory_id = ?",
                                (f"{reason} [trace {source_trace_id}]", memory_id))
            if not cursor.rowcount:
                raise ValueError("Unknown memory ID")
            db.commit()
