from langgraph.graph import StateGraph, END
from agent.state import AgentState
from agent.nodes import analyzer_node, gap_detector_node, retriever_node, recommendation_node, report_builder_node

def build_agent_graph():
    """Constructs and compiles the LangGraph agent workflow."""
    builder = StateGraph(AgentState)
    
    # Add all nodes
    builder.add_node("QuestionAnalyzer", analyzer_node)
    builder.add_node("GapDetector", gap_detector_node)
    builder.add_node("RAGRetriever", retriever_node)
    builder.add_node("RecommendationGenerator", recommendation_node)
    builder.add_node("ReportBuilder", report_builder_node)
    
    # Define the execution flow edges
    builder.set_entry_point("QuestionAnalyzer")
    builder.add_edge("QuestionAnalyzer", "GapDetector")
    builder.add_edge("GapDetector", "RAGRetriever")
    builder.add_edge("RAGRetriever", "RecommendationGenerator")
    builder.add_edge("RecommendationGenerator", "ReportBuilder")
    builder.add_edge("ReportBuilder", END)
    
    return builder.compile()

# The compiled graph ready to be invoked
agent_graph = build_agent_graph()
