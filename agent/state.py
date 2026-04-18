from typing import TypedDict, Dict, Any, List

class AgentState(TypedDict):
    """The shared state for the LangGraph agent."""
    question_data: Dict[str, Any]
    ml_results: Dict[str, str]
    analysis_stats: str
    learning_gaps: str
    rag_query: str
    rag_context: str
    recommendations: str
    final_report: Dict[str, Any]
