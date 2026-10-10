"""Session-scoped chat intake that invokes the same support graph as evaluations."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from .graph import SupportAgent

ID_PATTERN = re.compile(r"\bO-\d{3}\b", re.I)


@dataclass
class ChatResult:
    assistant_message: str
    status: str
    session_id: str
    trace_id: str | None = None
    recommended_action: str | None = None


class ConversationService:
    def __init__(self, agent: SupportAgent):
        self.agent = agent
        self.sessions: dict[str, dict] = {}

    def handle_message(self, session_id: str, customer_id_or_demo_identity: str, message: str,
                       optional_order_id: str | None = None, turn_id: str | None = None) -> ChatResult:
        if not session_id or not customer_id_or_demo_identity:
            raise ValueError("Session and fictional customer identity are required")
        session = self.sessions.setdefault(session_id, {"customer_id": customer_id_or_demo_identity,
            "messages": [], "known_order_id": None, "pending_order_id": False,
            "issue": "", "last_trace_id": None, "turns": {}})
        if session["customer_id"] != customer_id_or_demo_identity:
            raise ValueError("Session belongs to another fictional customer")
        turn_id = turn_id or str(uuid.uuid4())
        if turn_id in session["turns"]:
            return session["turns"][turn_id]
        ids = {match.upper() for match in ID_PATTERN.findall(message)}
        if optional_order_id:
            ids.add(optional_order_id.upper())
        malformed = bool(re.search(r"\bO-[A-Za-z0-9]+\b", message, re.I)) and not ids
        if malformed:
            result = ChatResult("Please check the order ID. It should look like O-123.",
                                "needs_order_id", session_id)
        elif len(ids) > 1:
            result = ChatResult("Please give me just one order ID for this request.", "needs_order_id", session_id)
        elif ids and (len(next(iter(ids))) != 5 or next(iter(ids)) not in self.agent.orders):
            result = ChatResult("I couldn't find that order ID in this demo. Please check the O-### number.",
                                "needs_order_id", session_id)
        else:
            order_id = next(iter(ids)) if ids else session["known_order_id"]
            if not order_id:
                if not session["issue"]:
                    session["issue"] = message.strip()
                session["pending_order_id"] = True
                result = ChatResult("Could you share the order ID (O-###) so I can check the current order?",
                                    "needs_order_id", session_id)
            else:
                changed = session["known_order_id"] not in (None, order_id)
                if changed:
                    session["issue"] = ""
                session["known_order_id"] = order_id
                session["pending_order_id"] = False
                issue = message.strip()
                if ID_PATTERN.fullmatch(issue) or (optional_order_id and not issue):
                    issue = session["issue"]
                elif session["issue"] and not changed:
                    issue = session["issue"] + " " + issue
                if not issue or ID_PATTERN.fullmatch(issue):
                    result = ChatResult("What happened with this order?", "needs_issue", session_id)
                else:
                    session["issue"] = issue
                    case = {"id": f"CHAT-{turn_id}", "order_id": order_id, "message": issue,
                            "customer_id": customer_id_or_demo_identity}
                    try:
                        run = self.agent.run_case(case, None, version="chat", session_id=session_id,
                                                  customer_id=customer_id_or_demo_identity,
                                                  messages=session["messages"] + [{"role": "user", "content": message}])
                        session["last_trace_id"] = run["trace_id"]
                        status = run["status"]
                        result = ChatResult(run["resolution"]["customer_reply"], status, session_id,
                                            run["trace_id"], run["resolution"]["action"])
                    except Exception:
                        result = ChatResult("I couldn't complete the review. Please contact a support specialist.",
                                            "error", session_id)
        session["messages"].extend([{"role": "user", "content": message},
                                    {"role": "assistant", "content": result.assistant_message}])
        session["turns"][turn_id] = result
        return result
