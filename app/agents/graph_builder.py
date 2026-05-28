import logging
from langgraph.graph import StateGraph, START, END
from app.agents.retrieval_agent import AgentState, retrieve_node, generate_node
from app.agents.clinical_agent import clinical_router_node, route_after_router
from app.agents.summarizer_agent import run_soap_node
from app.tools.patient_timeline_tool import run_timeline_tool
from app.tools.medication_checker_tool import run_medication_tool
from app.agents.safety_wrapper import apply_safety_wrapper

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------
# COMBINED NODES
# Each tool needs context first — retrieve then run the tool.
# ----------------------------------------------------------------

async def retrieve_then_generate(state: AgentState) -> AgentState:
    """Default RAG path — retrieve chunks then generate answer."""
    state = await retrieve_node(state)
    state = await generate_node(state)
    state["reply"] = apply_safety_wrapper(state["reply"], state["context"])
    return state


async def retrieve_then_timeline(state: AgentState) -> AgentState:
    """Retrieve relevant chunks then extract timeline."""
    state = await retrieve_node(state)
    state = await run_timeline_tool(state)
    state["reply"] = apply_safety_wrapper(state["reply"], state["context"])
    return state


async def retrieve_then_medication(state: AgentState) -> AgentState:
    """Retrieve relevant chunks then extract medication list."""
    state = await retrieve_node(state)
    state = await run_medication_tool(state)
    state["reply"] = apply_safety_wrapper(state["reply"], state["context"])
    return state


async def soap_with_safety(state: AgentState) -> AgentState:
    """Generate SOAP note then run silent safety check."""
    state = await run_soap_node(state)
    state["reply"] = apply_safety_wrapper(state["reply"], state["context"])
    return state


# ----------------------------------------------------------------
# BUILD GRAPH
# ----------------------------------------------------------------

def build_rag_graph():
    """
    Assemble the clinical LangGraph.

    Flow:
    START → clinical_router
                ↓ (conditional edge based on intent)
        ┌───────┼────────────┬──────────┐
        ↓       ↓            ↓          ↓
    retrieve  timeline  medication   soap
        ↓       ↓            ↓          ↓
                        END

    Each branch runs retrieve first (except soap which
    does its own fetching), then the specialist tool,
    then the silent safety check.
    """
    graph = StateGraph(AgentState)

    graph.add_node("clinical_router", clinical_router_node)
    graph.add_node("retrieve", retrieve_then_generate)
    graph.add_node("timeline", retrieve_then_timeline)
    graph.add_node("medication", retrieve_then_medication)
    graph.add_node("soap", soap_with_safety)

    graph.add_edge(START, "clinical_router")

    graph.add_conditional_edges(
        "clinical_router",
        route_after_router,
        {
            "retrieve":   "retrieve",
            "timeline":   "timeline",
            "medication": "medication",
            "soap":       "soap",
        }
    )

    graph.add_edge("retrieve",   END)
    graph.add_edge("timeline",   END)
    graph.add_edge("medication", END)
    graph.add_edge("soap",       END)

    compiled = graph.compile()
    logger.info("Graph compiled: START → router → [retrieve|timeline|medication|soap] → END")
    return compiled


async def run_rag_graph(
    message: str,
    patient_id: str,
    session_id: str = "default",
) -> dict:
    """Run the clinical graph and return the final state."""
    graph = build_rag_graph()

    initial_state: AgentState = {
        "message": message,
        "patient_id": patient_id,
        "session_id": session_id,
        "context": "",
        "sources": [],
        "reply": "",
        "error": "",
        "intent": "",
    }

    logger.info(f"Graph start — patient: {patient_id} | '{message[:60]}'")
    final_state = await graph.ainvoke(initial_state)
    logger.info(f"Graph end — intent: {final_state.get('intent')}")
    return final_state