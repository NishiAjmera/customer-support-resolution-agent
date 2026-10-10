"""Explicit LangGraph for planning, tool use, and resolution."""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langsmith.run_helpers import trace
from pydantic import BaseModel

from .memory import instruction_with_rules
from .prompts import Plan, plan_prompt, resolution_prompt
from .tools import create_resolution, get_order_details, load_records, search_policy


class AgentState(TypedDict, total=False):
    case: dict
    instruction: str
    steps: list[dict]
    plan: dict
    order: dict
    policies: list[dict]
    resolution: dict


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

        with trace("gemini-generate", run_type="llm", inputs={"prompt": prompt, "schema": schema.__name__},
                   metadata={"model": self.name}) as run:
            response = self.client.models.generate_content(
                model=self.name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                    response_schema=schema,
                ),
            )
            if not response.text:
                raise RuntimeError("Gemini returned no text response")
            result = schema.model_validate_json(response.text)
            run.end(outputs=result.model_dump())
            return result


def build_graph(model, orders: dict[str, dict], policies: dict[str, dict], store=None):
    """Compile the four-node support workflow."""

    def plan_tools(state: AgentState) -> dict:
        case = state["case"]
        catalog = [{"id": p["id"], "title": p["title"]} for p in policies.values()]
        plan = model.generate(plan_prompt(state["instruction"], case, catalog), Plan).model_dump()
        if plan["order_id"] != case["order_id"]:
            raise ValueError(f"Planner selected wrong order for {case['id']}")
        return {"plan": plan, "steps": state["steps"] + [{"type": "plan", "output": plan}]}

    def order_node(state: AgentState) -> dict:
        order = get_order_details(state["plan"]["order_id"], orders)
        return {"order": order, "steps": state["steps"] + [
            {"type": "tool", "name": "get_order_details", "input": state["plan"]["order_id"], "output": order}
        ]}

    def policy_node(state: AgentState) -> dict:
        found = search_policy(state["plan"]["policy_ids"], policies)
        return {"policies": found, "steps": state["steps"] + [
            {"type": "tool", "name": "search_policy", "input": state["plan"]["policy_ids"], "output": found}
        ]}

    def resolution_node(state: AgentState) -> dict:
        prompt = resolution_prompt(state["instruction"], state["case"], state["order"], state["policies"])
        result = create_resolution(model, prompt)
        return {"resolution": result, "steps": state["steps"] + [
            {"type": "tool", "name": "create_resolution", "output": result}
        ]}

    graph = StateGraph(AgentState)
    graph.add_node("plan_tools", plan_tools)
    graph.add_node("get_order_details", order_node)
    graph.add_node("search_policy", policy_node)
    graph.add_node("create_resolution", resolution_node)
    graph.add_edge(START, "plan_tools")
    graph.add_edge("plan_tools", "get_order_details")
    graph.add_edge("get_order_details", "search_policy")
    graph.add_edge("search_policy", "create_resolution")
    graph.add_edge("create_resolution", END)
    return graph.compile(store=store)


class SupportAgent:
    def __init__(self, model, memory=None):
        self.model = model
        self.memory = memory
        self.orders = load_records("orders.json")
        self.policies = load_records("policies.json")
        self.graph = build_graph(model, self.orders, self.policies,
                                 store=memory.graph_store if memory else None)

    def run_case(self, case: dict, rules: list[str] | None = None, version: str = "baseline") -> dict:
        if rules is None:
            rules = self.memory.active_rules() if self.memory else []
        result = self.graph.invoke({
            "case": case,
            "instruction": instruction_with_rules(rules),
            "steps": [],
        }, config={"metadata": {"case_id": case["id"], "version": version}})
        return {"case_id": case["id"], "version": version, "steps": result["steps"]}
