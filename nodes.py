import os
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from state import AgentState
from rag import PedagogyRAG

# Initialize RAG singleton
rag_system = PedagogyRAG()

def get_llm():
    """Initializes the LLM specifically with Groq's Llama3.1 for the agent."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set. Please update your .env file.")
    return ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.2, groq_api_key=api_key)

def analyzer_node(state: AgentState) -> AgentState:
    qd = state.get("question_data", {})
    ml = state.get("ml_results", {})
    
    stats = f"Question Theme: {qd.get('subject', 'Unknown')} - {qd.get('topic', 'Unknown')}. "
    stats += f"Predicted Bloom Level: {ml.get('bloom_level', 'Unknown')}. "
    stats += f"Predicted Difficulty: {ml.get('difficulty', 'Unknown')}. "
    stats += f"Performance: Average score {qd.get('avg_score', 0)}, with {qd.get('correct_percentage', 0)}% correct from {qd.get('num_students_attempted', 0)} attempts."
    
    state["analysis_stats"] = stats
    return state

def gap_detector_node(state: AgentState) -> AgentState:
    qd = state.get("question_data", {})
    correct_pct = qd.get('correct_percentage', 0)
    time_taken = qd.get('time_taken_minutes', 0)
    
    gaps = []
    if correct_pct < 40:
        gaps.append("High Failure Rate metric indicates a severe foundational gap or potentially a poorly phrased question.")
    elif correct_pct < 70:
        gaps.append("Moderate Gap metric points to partial understanding; students may struggle with multi-step processes or distractors.")
        
    if time_taken > 3.0:
        gaps.append("High Cognitive Load is predicted as the time taken suggests the question involves reading comprehension hurdles.")
        
    state["learning_gaps"] = " ".join(gaps) if gaps else "No major gaps. Metrics indicate steady comprehension."
    
    # Formulate query for RAG
    bloom = state.get("ml_results", {}).get("bloom_level", "Remember")
    state["rag_query"] = f"Bloom's {bloom} level questions and addressing gaps like: {state['learning_gaps']}"
    return state

def retriever_node(state: AgentState) -> AgentState:
    query = state.get("rag_query", "Pedagogical best practices")
    context = rag_system.retrieve_guidelines(query)
    state["rag_context"] = context
    return state

def recommendation_node(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = PromptTemplate.from_template(
        "You are an expert pedagogical AI assistant. Analyze the following question's metrics and learning gaps, "
        "and provide actionable recommendations for an educator to improve it, using the provided pedagogical context.\n\n"
        "Question Statistics: {stats}\n\n"
        "Detected Learning Gaps: {gaps}\n\n"
        "Pedagogical References (RAG): {context}\n\n"
        "Provide 3 specific and actionable recommendations. Write in a natural, conversational tone using plain paragraphs. Do NOT use any markdown formatting, bullet points, asterisks, bold text, or hashtags."
    )
    chain = prompt | llm
    
    try:
        result = chain.invoke({
            "stats": state.get("analysis_stats", ""),
            "gaps": state.get("learning_gaps", ""),
            "context": state.get("rag_context", "")
        })
        state["recommendations"] = result.content
    except Exception as e:
        state["recommendations"] = f"Error generating recommendations: {str(e)}"
        
    return state

def report_builder_node(state: AgentState) -> AgentState:
    state["final_report"] = {
        "Assessment Quality Summary": state.get("analysis_stats", ""),
        "Difficulty Distribution Analysis": state.get("ml_results", {}).get("difficulty", "Unknown"),
        "Identified Learning Gaps": state.get("learning_gaps", ""),
        "Recommended Improvements": state.get("recommendations", ""),
        "Pedagogical References": "Guidelines retrieved systematically using FAISS Vector Search.",
        "Ethical/Educational Disclaimer": "This analysis is AI-generated for advisory purposes. Final decisions should rely on an educator's professional judgment."
    }
    return state
