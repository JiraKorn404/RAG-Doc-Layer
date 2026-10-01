"""Decide whether the question needs the documents."""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from ragdoc.agent import prompts
from ragdoc.agent.context import history_messages, invoke_decision
from ragdoc.agent.state import AgentDeps, AgentState, Node, NodeResult


class RouteDecision(BaseModel):
    reason: str
    route: Literal["retrieve", "direct"]


def make_route(deps: AgentDeps) -> Node:
    def route(state: AgentState) -> NodeResult:
        messages = [
            SystemMessage(prompts.ROUTE_SYSTEM),
            *history_messages(state.get("history", [])),
            HumanMessage(state["question"]),
        ]
        choice, reason, usage = invoke_decision(
            deps.control_model, messages, label="Route", choices=("retrieve", "direct")
        )
        if choice is None:
            # Searching when it was not needed costs time; skipping a needed search costs the
            # answer. So an unreadable decision means: search.
            choice, reason = "retrieve", "Could not read the routing decision"
        decision = RouteDecision(reason=reason, route=choice)

        action = "Search the documents" if decision.route == "retrieve" else "Answer directly"
        return NodeResult(
            summary=f"{action}: {decision.reason}",
            update={"route": decision.route, "retries": 0, "usage": usage},
            payload={"route": decision.route, "reason": decision.reason},
        )

    return route
