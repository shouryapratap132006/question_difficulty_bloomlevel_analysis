import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

from logistic_regression_deployment import BloomModelDeployer
from graph import agent_graph

# Page configuration
st.set_page_config(
    page_title="Intelligent Assessment System",
    page_icon="🎓",
    layout="wide"
)

# Title and Description
st.title("🎓 Intelligent Assessment Design System")
st.markdown("Classify educational questions and get AI-assisted pedagogical recommendations using natural prose.")

# Initialize the model worker
@st.cache_resource
def get_worker():
    worker = BloomModelDeployer()
    worker.load_models()
    return worker

worker = get_worker()

# Sidebar Status & Settings
with st.sidebar:
    st.header("Admin Settings")
    if not worker.bloom_model:
        st.error("⚠️ Models not loaded.")
        if st.button("Train Model Now"):
            with st.spinner("Training..."):
                worker.train("final.csv")
                st.success("Training complete!")
                st.rerun()
    else:
        st.success("✅ ML Models Ready")
        
    api_key_status = "✅ Configured" if os.getenv("GROQ_API_KEY") else "❌ Missing"
    st.markdown(f"**Groq API Key:** {api_key_status}")
    
    if st.button("Force Retrain Models"):
        with st.spinner("Training..."):
            worker.train("final.csv")
            st.success("Re-training successful!")
            st.rerun()

# Unified Form inputs
st.subheader("📝 Enter Question Details")
with st.container(border=True):
    question_text = st.text_area("Question Text", placeholder="e.g., Explain the process of photosynthesis in your own words.")
    col1, col2, col3 = st.columns(3)
    with col1:
        subject = st.text_input("Subject", "Science")
        avg_score = st.number_input("Avg Score (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    with col2:
        topic = st.text_input("Topic", "Biology")
        correct_pct = st.number_input("Correct % (0-100)", min_value=0, max_value=100, value=70)
    with col3:
        attempted = st.number_input("Total Attempts", min_value=1, value=100)
        correct = st.number_input("Total Correct", min_value=0, value=70)
        time_taken = st.number_input("Avg Time (min)", min_value=0.1, value=2.0)

sample_data = {
    "question_text": question_text,
    "subject": subject,
    "topic": topic,
    "avg_score": avg_score,
    "correct_percentage": float(correct_pct),
    "num_students_attempted": int(attempted),
    "num_students_correct": int(correct),
    "time_taken_minutes": float(time_taken)
}

st.markdown("---")

if st.button("Analyze & Generate Assessment Report", type="primary", use_container_width=True):
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY is missing from environment. Please add it to your .env file.")
    elif not question_text:
        st.warning("Please enter some question text.")
    elif not worker.bloom_model:
        st.error("Error: Models are not loaded.")
    else:
        st.subheader("📊 Output Report")
        
        # Step 1: ML Prediction
        with st.spinner("Evaluating ML metrics..."):
            try:
                results = worker.predict(sample_data)
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.info(f"Bloom's Taxonomy Level: {results['bloom_level']}")
                with col_res2:
                    st.success(f"Estimated Difficulty: {results['difficulty']}")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        # Step 2: Agent execution
        initial_state = {
            "question_data": sample_data,
            "ml_results": results
        }
        
        report = None
        try:
            with st.spinner("Generating AI pedagogical report..."):
                final_state = agent_graph.invoke(initial_state)
                report = final_state.get("final_report")
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            
        if report:
            st.markdown("---")
            st.header("✨ Comprehensive Assessment Report")
            st.markdown("Below is the specialized feedback generated collaboratively by the Machine Learning model and the Agentic Pedagogical AI.")
            
            st.write("")
            
            # Use native Streamlit metrics
            st.subheader("📊 Performance & Classification")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Bloom's Level", results.get("bloom_level", "N/A"))
            m2.metric("Difficulty", results.get("difficulty", "N/A"))
            m3.metric("Average Score", f"{sample_data.get('avg_score', 0):.2f}")
            m4.metric("Correct Rate", f"{sample_data.get('correct_percentage', 0)}%")
            
            st.write("")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("🧠 Assessment Summary")
                st.info(report['Assessment Quality Summary'], icon="ℹ️")
                
            with col_right:
                st.subheader("⚠️ Learning Gaps")
                # Warning color is best for gaps
                st.warning(report['Identified Learning Gaps'], icon="🚨")
                
            st.write("")
            
            st.subheader("💡 Pedagogical AI Recommendations")
            with st.chat_message("ai"):
                st.write("Based on my analysis of the student metrics and educational best practices, here are my suggestions to improve this question:")
                st.write(report['Recommended Improvements'])
                
            st.write("")
            
            with st.expander("📚 View Retrieved Pedagogical References (RAG)"):
                st.caption(report['Pedagogical References'])
                
            st.divider()
            st.caption(report['Ethical/Educational Disclaimer'])
