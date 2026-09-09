import os
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from app.vlm_engine import analyze_drone_frame
from app.pinecone_rag import query_sop_database

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Aerial-Perimeter-Sentinel"

class SentinelState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    image_data: bytes | None
    route: str
    verified: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def router_node(state: SentinelState) -> SentinelState:
    last_msg = state["messages"][-1].content
    has_image = state.get("image_data") is not None
    
    prompt = f"""Classify incoming drone telemetry request:
    Image Attached: {has_image}
    Query: {last_msg}
    
    Respond with ONE WORD only:
    - 'vision': Image is provided and requires inspection.
    - 'sop': Request asks about protocols, regulations, or rules.
    - 'direct': Simple request or conversation.
    """
    decision = llm.invoke([HumanMessage(content=prompt)]).content.strip().lower()
    return {"route": decision}

def vision_node(state: SentinelState) -> SentinelState:
    query = state["messages"][-1].content
    img_bytes = state.get("image_data")
    if not img_bytes:
        return {"messages": [AIMessage(content="Error: Image processing requested, but no image was attached.")]}
    
    result = analyze_drone_frame(img_bytes, query)
    return {"messages": [AIMessage(content=result)]}

def sop_node(state: SentinelState) -> SentinelState:
    query = state["messages"][-1].content
    result = query_sop_database(query)
    return {"messages": [AIMessage(content=result)]}

def verifier_node(state: SentinelState) -> SentinelState:
    response_text = state["messages"][-1].content
    prompt = f"Verify if this operational output contains actionable analysis:\n{response_text}\nReply 'YES' or 'NO'."
    res = llm.invoke([HumanMessage(content=prompt)]).content.strip().upper()
    return {"verified": "YES" in res}

def route_decision(state: SentinelState) -> str:
    return state["route"]

workflow = StateGraph(SentinelState)
workflow.add_node("router", router_node)
workflow.add_node("vision", vision_node)
workflow.add_node("sop", sop_node)
workflow.add_node("verifier", verifier_node)

workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "vision": "vision",
        "sop": "sop",
        "direct": END
    }
)
workflow.add_edge("vision", "verifier")
workflow.add_edge("sop", "verifier")
workflow.add_edge("verifier", END)

sentinel_agent = workflow.compile()
