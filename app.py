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
        
        status_container = st.container(border=True)
        with status_container:
            st.markdown("**Agent Execution Progress**")
            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            step4 = st.empty()
            step5 = st.empty()
            
        report = None
        try:
            for event in agent_graph.stream(initial_state):
                if "QuestionAnalyzer" in event:
                    step1.text("Step 1: Analyzed Question & ML Results")
                elif "GapDetector" in event:
                    step2.text("Step 2: Detected Student Learning Gaps")
                elif "RAGRetriever" in event:
                    step3.text("Step 3: Retrieved Pedagogical References")
                elif "RecommendationGenerator" in event:
                    step4.text("Step 4: Generated AI Recommendations")
                elif "ReportBuilder" in event:
                    step5.text("Step 5: Built Final Report")
                    report = event["ReportBuilder"]["final_report"]
        except Exception as e:
            st.error(f"Agent execution failed: {e}")
            
        if report:
            # Display plain text using natural prose formatting
            st.markdown("### Final Structured Report")
            
            with st.container(border=True):
                st.markdown("#### Assessment Quality Summary")
                st.write(report['Assessment Quality Summary'])
                
            with st.container(border=True):
                st.markdown("#### Identified Learning Gaps")
                st.write(report['Identified Learning Gaps'])
                
            with st.container(border=True):
                st.markdown("#### Recommended Improvements")
                st.write(report['Recommended Improvements'])
                
            with st.container(border=True):
                st.markdown("#### Pedagogical References")
                st.write(report['Pedagogical References'])
                
            st.caption(report['Ethical/Educational Disclaimer'])
