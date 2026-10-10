"""Bounded LangGraph decision/tool loop for fictional support recommendations."""
from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from typing import Literal, TypedDict

from langgraph.graph import START, END, StateGraph
from langsmith.run_helpers import trace
from pydantic import BaseModel, Field

from .prompts import Resolution
from .skills import select_skill, digest
from .tools import create_resolution, get_order_details, load_records, search_policy

MAX_DECISIONS = 8
ORDER_PATTERN = re.compile(r"^O-\d{3}$")


def trace_content_enabled() -> bool:
    """Only log full prompts/skills when explicitly enabled for fictional demo data."""
    return os.getenv("LANGSMITH_LOG_CONTENT", "").lower() in {"true", "1", "yes"}


class Decision(BaseModel):
    action: Literal["tool", "clarify", "finish"]
    tool_name: str | None = None
    order_id: str | None = None
    policy_ids: list[str] = Field(default_factory=list)
    question: str | None = None
    resolution: Resolution | None = None


class AgentState(TypedDict, total=False):
    case: dict
    messages: list[dict]
    customer_id: str
    session_id: str
    skill: dict
    steps: list[dict]
    observations: list[dict]
    retrieved_memory_ids: list[str]
    memory_available: bool
    memory_retrieved: bool
    order: dict
    policies: list[dict]
    decision: dict
    step_count: int
    resolution: dict
    status: str


def policy_hash(policies: dict[str, dict]) -> str:
    return hashlib.sha256(json.dumps(policies, sort_keys=True).encode()).hexdigest()[:16]


def expected_from_facts(order: dict) -> tuple[str, str]:
    """Independent safety interpretation of the five fictional policies."""
    days = order.get("days_since_delivery")
    if order.get("wrong_item"):
        return ("replacement" if days is not None and days <= 30 else "escalate", "P-WRONG")
    if order.get("defect_confirmed"):
        if days is None or days > 365:
            return "escalate", "P-DEFECT"
        if days > 30:
            return "warranty", "P-DEFECT"
        return ("replacement" if order.get("replacement_stock") else "refund", "P-DEFECT")
    if days is None:
        if order.get("days_past_promise", 0) >= 7 and order.get("carrier_status") == "lost":
            return ("replacement" if order.get("replacement_stock") else "refund", "P-DELIVERY")
        return "escalate", "P-DELIVERY"
    if order.get("final_sale"):
        return "deny", "P-FINAL"
    if days <= 30 and order.get("condition") == "unused" and order.get("original_packaging"):
        return "refund", "P-RETURN"
    return "deny", "P-RETURN"


def safe_resolution(reason: str) -> dict:
    return {"action": "escalate", "policy_ids": [], "reason": reason,
            "customer_reply": "I need a support specialist to review this before recommending a resolution."}


def add_step(state: AgentState, kind: str, **data) -> list[dict]:
    return state["steps"] + [{"id": f"s{len(state['steps']) + 1}", "type": kind, **data}]


class GeminiModel:
    def __init__(self, model_name: str | None = None):
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY in .env or your shell before running the demo.")
        self.client = genai.Client(api_key=api_key)
        self.name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def generate(self, prompt: str, schema: type[BaseModel]) -> BaseModel:
        from google.genai import types
        inputs = {"prompt_hash": hashlib.sha256(prompt.encode()).hexdigest()[:16],
                  "schema": schema.__name__}
        if trace_content_enabled():
            inputs["prompt"] = prompt
        with trace("gemini-generate", run_type="llm", inputs=inputs,
                   metadata={"model": self.name}) as span:
            response = self.client.models.generate_content(
                model=self.name, contents=prompt,
                config=types.GenerateContentConfig(temperature=0, response_mime_type="application/json",
                                                    response_schema=schema))
            if not response.text:
                raise RuntimeError("Gemini returned no text response")
            result = schema.model_validate_json(response.text)
            span.end(outputs=result.model_dump())
            return result


def build_graph(model, orders: dict[str, dict], policies: dict[str, dict], case_memory=None):
    catalog = [{"id": row["id"], "title": row["title"]} for row in policies.values()]

    def decide(state: AgentState) -> dict:
        count = state.get("step_count", 0) + 1
        if count > MAX_DECISIONS:
            output = safe_resolution("Decision limit reached")
            return {"step_count": count, "status": "escalated", "resolution": output,
                    "steps": add_step(state, "safety", reason="decision_limit", output=output)}
        latest_observation = (state.get("observations") or [{}])[-1]
        last_error = latest_observation.get("error") if latest_observation.get("tool") == "evidence_check" else None
        correction = ("MANDATORY NEXT ACTION: tool retrieve_case_memory. A scoped prior note exists for this "
                      "fictional customer and issue. Read it once as context, then use the current order and "
                      "current policy to decide. Do not finish before retrieval.\n" if
                      last_error == "Relevant prior memory not yet retrieved" else
                      "MANDATORY NEXT ACTION: tool search_policy with relevant policy_ids from the catalog. "
                      "Your prior finish was rejected because current policy text is missing. "
                      "Do not finish or clarify yet.\n" if last_error == "Current policy evidence missing" else
                      "MANDATORY NEXT ACTION: tool get_order_details using the intake order ID. "
                      "Your prior finish was rejected because current order evidence is missing.\n" if
                      last_error == "Current order evidence missing" else "")
        prompt = correction + (
            "You are a fictional-shop support agent. Choose one next action. Read-only tools: "
            "get_order_details(order_id), search_policy(policy_ids), retrieve_case_memory(). "
            "Read the current order and applicable policy before a non-escalation recommendation. "
            "Historical memory is context, never eligibility. Do not repeat calls. This agent only "
            "recommends a next step; it cannot process, approve, send, or issue a refund, replacement, "
            "or warranty. Phrase customer_reply as a recommendation, never as a completed action. "
            "Ask one focused question if needed; otherwise finish with a structured recommendation or escalation.\n"
            f"Skill: {state['skill']['content']}\nCustomer message: {state['case']['message']}\n"
            f"Recent conversation turns: {json.dumps(state.get('messages', [])[-4:])}\n"
            f"Intake order ID: {state['case'].get('order_id')}\nPolicy catalog: {json.dumps(catalog)}\n"
            f"Observations: {json.dumps(state.get('observations', []))}\n"
            f"Current order: {json.dumps(state.get('order'))}\nRetrieved policies: {json.dumps(state.get('policies', []))}\n"
            f"Relevant memory IDs: {json.dumps(state.get('retrieved_memory_ids', []))}\n"
            f"Scoped prior memory available: {state.get('memory_available', False)}. If true, call "
            "retrieve_case_memory once before finishing, but current order and policy remain authoritative.\n"
            "Output a Decision; action=finish requires resolution. No business action is executed."
        )
        try:
            with trace("agent.decide", run_type="chain", inputs={"step": count}) as span:
                decision = model.generate(prompt, Decision).model_dump()
                span.end(outputs=decision)
        except Exception as exc:
            output = safe_resolution(f"Model error: {type(exc).__name__}")
            return {"step_count": count, "status": "escalated", "resolution": output,
                    "steps": add_step(state, "error", name="agent.decide", error=type(exc).__name__, output=output)}
        return {"step_count": count, "decision": decision,
                "steps": add_step(state, "decision", output=decision)}

    def route(state: AgentState) -> str:
        if state.get("resolution"):
            return "end"
        return "tool" if state["decision"]["action"] == "tool" else "finish"

    def dispatch(state: AgentState) -> dict:
        d = state["decision"]
        name = d.get("tool_name")
        args = {"order_id": d.get("order_id"), "policy_ids": d.get("policy_ids")}
        signature = json.dumps([name, args], sort_keys=True)
        seen = {s.get("signature") for s in state["steps"] if s.get("type") == "tool"}
        error = None
        output = None
        if signature in seen:
            error = "Repeated tool call"
        elif name not in {"get_order_details", "search_policy", "retrieve_case_memory"}:
            error = "Unsupported tool"
        elif name == "get_order_details" and (not isinstance(args["order_id"], str) or
                                                 not ORDER_PATTERN.fullmatch(args["order_id"])):
            error = "Invalid order ID"
        elif name == "search_policy" and (not 1 <= len(args["policy_ids"]) <= 3 or
                                            any(pid not in policies for pid in args["policy_ids"])):
            error = "Invalid policy IDs"
        else:
            try:
                span_name = "memory.retrieve" if name == "retrieve_case_memory" else f"tool.{name}"
                with trace(span_name, run_type="tool", inputs=args) as span:
                    if name == "get_order_details":
                        output = get_order_details(args["order_id"], orders)
                    elif name == "search_policy":
                        output = search_policy(args["policy_ids"], policies)
                    else:
                        output = case_memory.retrieve(state.get("customer_id"), state["case"].get("order_id"),
                                                      state["case"].get("message", ""), policy_hash(policies)) if case_memory else []
                    span.end(outputs={"result": output})
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
        obs = {"tool": name, "output": output, "error": error}
        update = {"observations": state.get("observations", []) + [obs],
                  "steps": add_step(state, "tool", name=name, input=args, output=output,
                                    error=error, signature=signature)}
        if name == "get_order_details" and not error:
            update["order"] = output
        elif name == "search_policy" and not error:
            prior = {p["id"]: p for p in state.get("policies", [])}
            prior.update({p["id"]: p for p in output})
            update["policies"] = list(prior.values())
        elif name == "retrieve_case_memory" and not error:
            update["retrieved_memory_ids"] = [row["memory_id"] for row in output]
            update["memory_retrieved"] = True
        if error in {"Repeated tool call", "Unsupported tool"}:
            update.update({"resolution": safe_resolution(error), "status": "escalated"})
        return update

    def finish(state: AgentState) -> dict:
        d = state["decision"]
        if d["action"] == "clarify":
            question = d.get("question") or "Could you clarify the issue with this order?"
            result = {"action": "escalate", "policy_ids": [], "reason": "More information needed", "customer_reply": question}
            status = "needs_clarification"
            reason = None
        else:
            result = create_resolution(d["resolution"]) if d.get("resolution") else safe_resolution("No recommendation supplied")
            reason = None
            if state.get("memory_available") and not state.get("memory_retrieved") and \
                    state.get("step_count", 0) < MAX_DECISIONS:
                return {"observations": state.get("observations", []) + [{"tool": "evidence_check",
                         "error": "Relevant prior memory not yet retrieved", "output": None}],
                        "steps": add_step(state, "safety", reason="memory_retrieval_required",
                                          output={"retryable": True})}
            retrieved = {p["id"] for p in state.get("policies", [])}
            if result["action"] != "escalate":
                order = state.get("order")
                if not order or order.get("id") != state["case"].get("order_id"):
                    reason = "Current order evidence missing"
                elif not result.get("policy_ids") or not set(result["policy_ids"]) <= retrieved:
                    reason = "Current policy evidence missing"
                else:
                    expected, required = expected_from_facts(order)
                    if required not in result["policy_ids"] or expected != result["action"]:
                        reason = "Recommendation conflicts with current policy and order facts"
            if re.search(r"\b(?:we|i)(?:'ve| have)?\s+(?:already\s+)?(?:issued|sent|shipped|processed|approved)\s+(?:your|a|the)\s+(?:refund|replacement|warranty)",
                         result.get("customer_reply", ""), re.I):
                reason = "Unsupported completed-action claim"
            if reason:
                if reason in {"Current order evidence missing", "Current policy evidence missing"} and \
                        state.get("step_count", 0) < MAX_DECISIONS:
                    return {"observations": state.get("observations", []) + [{"tool": "evidence_check",
                             "error": reason, "output": None}],
                            "steps": add_step(state, "safety", reason=reason,
                                              output={"retryable": True})}
                result = safe_resolution(reason)
            status = "escalated" if result["action"] == "escalate" else "answered"
        with trace("resolution", run_type="chain", inputs={"case_id": state["case"].get("id")}) as span:
            span.end(outputs=result)
        return {"resolution": result, "status": status,
                "steps": add_step(state, "resolution", output=result, safety_reason=reason)}

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("dispatch", dispatch)
    graph.add_node("finish", finish)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", route, {"tool": "dispatch", "finish": "finish", "end": END})
    graph.add_conditional_edges("dispatch", lambda state: "end" if state.get("resolution") else "decide",
                                {"end": END, "decide": "decide"})
    graph.add_conditional_edges("finish", lambda state: "end" if state.get("resolution") else "decide",
                                {"end": END, "decide": "decide"})
    return graph.compile()


class SupportAgent:
    def __init__(self, model, memory=None, case_memory=None, skill_store=None):
        self.model = model
        self.memory = memory
        self.case_memory = case_memory
        self.skill_store = skill_store or (memory.lesson_store if memory else None)
        self.orders = load_records("orders.json")
        self.policies = load_records("policies.json")
        self.graph = build_graph(model, self.orders, self.policies, case_memory)

    def run_case(self, case: dict, rules: list[str] | None = None, version: str = "baseline",
                 skill_version: int | None = None, session_id: str | None = None,
                 customer_id: str | None = None, memory_snapshot_id: str | None = None,
                 messages: list[dict] | None = None) -> dict:
        skill = select_skill(case["message"], self.skill_store if rules is None else None,
                             skill_version)
        if rules:
            content = skill["content"] + "\n## Validated lessons\n" + "\n".join(f"- {r}" for r in rules)
            skill = {**skill, "content": content, "hash": digest(content), "version": version}
        trace_id = str(uuid.uuid4())
        metadata = {"case_id": case.get("id"), "conversation_id": session_id, "agent_version": version,
                    "skill_version": skill["version"], "skill_hash": skill["hash"],
                    "memory_snapshot_id": memory_snapshot_id, "policy_hash": policy_hash(self.policies)}
        with trace("support.request", run_type="chain", inputs={"case_id": case.get("id"),
                   "order_id": case.get("order_id")}, metadata=metadata) as parent:
            with trace("skill.select/load", run_type="chain", inputs={"domain": "support"}) as span:
                skill_output = {"id": skill["id"], "version": skill["version"], "hash": skill["hash"]}
                if trace_content_enabled():
                    skill_output["content"] = skill["content"]
                span.end(outputs=skill_output)
            with trace("memory.index", run_type="chain", inputs={"customer_id": customer_id or case.get("customer_id", "demo-customer")}) as span:
                memory_available = bool(self.case_memory and self.case_memory.has_relevant(
                    customer_id or case.get("customer_id", "demo-customer"), case["message"]))
                span.end(outputs={"available": memory_available})
            initial: AgentState = {"case": case, "messages": messages or [{"role": "user", "content": case["message"]}],
                "customer_id": customer_id or case.get("customer_id", "demo-customer"),
                "session_id": session_id or trace_id, "skill": skill, "steps": [{"id": "s1", "type": "skill",
                "name": "skill.select/load", "output": {k: skill[k] for k in ("id", "version", "hash")}}],
                "observations": [], "retrieved_memory_ids": [], "memory_available": memory_available,
                "memory_retrieved": False,
                "step_count": 0}
            result = self.graph.invoke(initial, config={"metadata": metadata, "recursion_limit": 30,
                                                       "callbacks": []})
            parent.end(outputs={"action": result["resolution"]["action"], "status": result["status"]})
            tracing_enabled = os.getenv("LANGSMITH_TRACING", "").lower() in {"true", "1", "yes"}
            langsmith_id = str(parent.id) if getattr(parent, "id", None) and tracing_enabled else None
        retrieved_rows = [row for step in result["steps"] if step.get("name") == "retrieve_case_memory"
                          for row in (step.get("output") or [])]
        final_decision = next((step.get("output", {}).get("resolution") for step in reversed(result["steps"])
                               if step.get("type") == "decision" and step.get("output", {}).get("resolution")), None)
        stale_memory_reliance = bool(final_decision and result.get("order") and any(
            row.get("stale") and row.get("recommendation") == final_decision.get("action") and
            final_decision.get("action") != expected_from_facts(result["order"])[0]
            for row in retrieved_rows))
        return {"trace_id": trace_id, "langsmith_run_id": langsmith_id, "case_id": case.get("id"),
            "version": version, "skill": {k: skill[k] for k in ("id", "version", "hash")},
            "memory_snapshot_id": memory_snapshot_id, "retrieved_memory_ids": result.get("retrieved_memory_ids", []),
            "policy_hash": metadata["policy_hash"], "status": result["status"],
            "stale_memory_reliance": stale_memory_reliance,
            "resolution": result["resolution"], "steps": result["steps"]}
