"""Read active procedural memory into an agent instruction."""

from langgraph.store.memory import InMemoryStore

from .prompts import BASE_INSTRUCTION


class ProceduralMemory:
    """LangGraph store cache of the active SQLite lesson version."""

    def __init__(self, lesson_store):
        self.lesson_store = lesson_store
        self.graph_store = InMemoryStore()
        self.refresh()

    def refresh(self) -> None:
        active = self.lesson_store.active()
        self.graph_store.put(
            ("support", "procedural"), "active",
            {"version_id": active["id"] if active else None, "rules": active["rules"] if active else []},
        )

    def active_rules(self) -> list[str]:
        return self.graph_store.get(("support", "procedural"), "active").value["rules"]


def instruction_with_rules(rules: list[str]) -> str:
    if not rules:
        return BASE_INSTRUCTION
    return BASE_INSTRUCTION + "\nValidated lessons from prior cases:\n" + "\n".join(
        f"- {rule}" for rule in rules
    )
