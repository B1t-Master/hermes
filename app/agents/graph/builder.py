"""LangGraph construction and the shared compiled graph singleton."""

from langgraph.graph import END, StateGraph

from .nodes.analysis import analyse_sentiment, classify
from .nodes.answer import answer
from .nodes.escalation import escalate
from .nodes.greeting import dispatch, greet
from .nodes.retrieval import baggage_lookup, flight_lookup, retrieve
from .nodes.route import route
from .routing import dispatch_next, route_next
from .state import AgentState

_graph = None


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("dispatch", dispatch)
    graph.add_node("greet", greet)
    graph.add_node("analyse_sentiment", analyse_sentiment)
    graph.add_node("classify", classify)
    graph.add_node("route", route)
    graph.add_node("retrieve", retrieve)
    graph.add_node("flight_lookup", flight_lookup)
    graph.add_node("baggage_lookup", baggage_lookup)
    graph.add_node("generate_answer", answer)
    graph.add_node("handle_escalation", escalate)

    graph.set_entry_point("dispatch")
    graph.add_conditional_edges(
        "dispatch",
        dispatch_next,
        {"greeting": "greet", "chat": "analyse_sentiment"},
    )
    graph.add_edge("greet", END)
    graph.add_edge("analyse_sentiment", "classify")
    graph.add_edge("classify", "route")
    graph.add_conditional_edges(
        "route",
        route_next,
        {
            "escalate": "handle_escalation",
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("retrieve", "flight_lookup")
    graph.add_edge("flight_lookup", "baggage_lookup")
    graph.add_edge("baggage_lookup", "generate_answer")
    graph.add_edge("generate_answer", END)
    graph.add_edge("handle_escalation", END)
    return graph.compile()


def get_graph():
    """Shared compiled graph."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
