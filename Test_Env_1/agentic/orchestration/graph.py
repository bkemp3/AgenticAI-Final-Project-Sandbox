from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agentic.orchestration.nodes import (
    compile_tree,
    execute_tree_node,
    generate_plan,
    handle_error,
    select_planner,
    visualize_tree,
)
from agentic.orchestration.state import OrchestrationState


def build_orchestration_app():
    """Build the LangGraph workflow for the behavior tree lifecycle."""

    graph = StateGraph(OrchestrationState)
    graph.add_node("select_planner", select_planner)
    graph.add_node("generate_plan", generate_plan)
    graph.add_node("compile_tree", compile_tree)
    graph.add_node("visualize_tree", visualize_tree)
    graph.add_node("execute_tree", execute_tree_node)
    graph.add_node("handle_error", handle_error)

    graph.add_edge(START, "select_planner")
    graph.add_conditional_edges(
        "select_planner",
        _route_after_select_planner,
        {"next": "generate_plan", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "generate_plan",
        _route_after_generate_plan,
        {"next": "compile_tree", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "compile_tree",
        _route_after_compile_tree,
        {"next": "visualize_tree", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "visualize_tree",
        _route_after_visualize_tree,
        {"next": "execute_tree", "handle_error": "handle_error"},
    )
    graph.add_conditional_edges(
        "execute_tree",
        _route_after_execution,
        {"end": END, "handle_error": "handle_error"},
    )
    graph.add_edge("handle_error", END)

    return graph.compile()


def _route_after_select_planner(state: OrchestrationState) -> str:
    return "handle_error" if state.get("error_message") else "next"


def _route_after_generate_plan(state: OrchestrationState) -> str:
    return "handle_error" if state.get("error_message") else "next"


def _route_after_compile_tree(state: OrchestrationState) -> str:
    return "handle_error" if state.get("error_message") else "next"


def _route_after_visualize_tree(state: OrchestrationState) -> str:
    return "handle_error" if state.get("error_message") else "next"


def _route_after_execution(state: OrchestrationState) -> str:
    if state.get("error_message"):
        return "handle_error"
    return "end"
