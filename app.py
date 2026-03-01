import streamlit as st
import pandas as pd
from logistic_regression_deployment import BloomModelDeployer

# Page configuration
st.set_page_config(
    page_title="Bloom Level & Difficulty Classifier",
    page_icon="🎓",
    layout="centered"
)

# Custom CSS for a premium look
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border: none;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        border-left: 5px solid #4CAF50;
    }
    .header-style {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1 class='header-style'>🎓 Question Classification System</h1>", unsafe_allow_html=True)
st.markdown("""
    <p style='color: #64748B;'>Classify educational questions into Bloom's Taxonomy and Difficulty levels using machine learning.</p>
""", unsafe_allow_html=True)

# Initialize the model worker
@st.cache_resource
def get_worker():
    worker = BloomModelDeployer()
    worker.load_models()
    return worker

worker = get_worker()

# Status Check
if not worker.bloom_model:
    st.error("⚠️ Models not found in the 'models/' directory. Please ensure the model is trained.")
    if st.button("Train Model Now"):
        with st.spinner("Training..."):
            worker.train("final.csv")
            st.success("Training complete!")
            st.rerun()
else:
    st.info("💡 Model is ready for prediction.")

# Form Layout
with st.container():
    st.subheader("1. Question Details")
    question_text = st.text_area("Enter your question text here", placeholder="e.g., Explain the process of photosynthesis in your own words.")
    
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        subject = st.text_input("Subject", "Science")
    with col_meta2:
        topic = st.text_input("Topic", "Biology")

st.markdown("---")

with st.container():
    st.subheader("2. Performance Metrics")
    col_met1, col_met2, col_met3 = st.columns(3)
    
    with col_met1:
        avg_score = st.number_input("Avg Score (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
        correct_pct = st.number_input("Correct % (0-100)", min_value=0, max_value=100, value=70)
    
    with col_met2:
        attempted = st.number_input("Total Attempts", min_value=1, value=100)
        correct = st.number_input("Total Correct", min_value=0, value=70)
        
    with col_met3:
        time_taken = st.number_input("Avg Time (min)", min_value=0.1, value=2.0)

# Prediction Logic
if st.button("Classify Question"):
    if not question_text:
        st.warning("Please enter some question text.")
    elif not worker.bloom_model:
        st.error("Error: Models are not loaded.")
    else:
        with st.spinner("Analyzing..."):
            sample = {
                "question_text": question_text,
                "subject": subject,
                "topic": topic,
                "avg_score": avg_score,
                "correct_percentage": float(correct_pct),
                "num_students_attempted": int(attempted),
                "num_students_correct": int(correct),
                "time_taken_minutes": float(time_taken)
            }
            
            try:
                results = worker.predict(sample)
                
                st.markdown("---")
                st.subheader("3. Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    st.markdown(f"""
                        <div class="prediction-card">
                            <p style='color: #64748B; margin-bottom: 5px;'>Bloom's Level</p>
                            <h3 style='color: #1E3A8A; margin-top: 0;'>{results['bloom_level']}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                
                with res_col2:
                    st.markdown(f"""
                        <div class="prediction-card" style="border-left-color: #3B82F6;">
                            <p style='color: #64748B; margin-bottom: 5px;'>Difficulty</p>
                            <h3 style='color: #1E3A8A; margin-top: 0;'>{results['difficulty']}</h3>
                        </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# Sidebar for advanced options
with st.sidebar:
    st.header("Admin Settings")
    if st.checkbox("Show Raw JSON Output"):
        if 'results' in locals():
            st.json(results)
    
    if st.button("Force Retrain Models"):
        with st.spinner("Training..."):
            worker.train("final.csv")
            st.success("Re-training successful!")
            st.rerun()
