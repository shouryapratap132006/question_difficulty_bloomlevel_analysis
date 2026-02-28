import streamlit as st
import pandas as pd
from logistic_regression_deployment import BloomModelDeployer

# Title of the app
st.title("Question Classifier App")

# Header and Subheader
st.header("Predict Bloom Level and Difficulty")
st.subheader("Enter the details of your question below")

# Initialize the model worker
worker = BloomModelDeployer()
if worker.load_models():
    st.success("Model is ready for prediction!")
else:
    st.warning("Model not found. Please train it first.")

# Question Text Input
question_text = st.text_area("Enter your Question Text here")

# Subject and Topic
subject = st.text_input("Enter Subject", "General Knowledge")
topic = st.text_input("Enter Topic", "Geography")

# Numerical Metrics
st.info("Enter student performance data")
avg_score = st.number_input("Average Score (0.0 to 1.0)", value=0.7)
correct_pct = st.number_input("Correct Percentage (0 to 100)", value=70)
attempted = st.number_input("Number of Students who Attempted", value=100)
correct = st.number_input("Number of Students who got it Correct", value=70)
time_taken = st.number_input("Average Time Taken (minutes)", value=2.0)

# Prediction Button
if st.button("Predict"):
    if not question_text:
        st.error("Please fill in the question text!")
    else:
        # Prepare the data dictionary
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
        
        # Get result
        results = worker.predict(sample)
        
        # Display results simply
        st.success(f"Predicted Bloom Level: {results['bloom_level']}")
        st.success(f"Predicted Difficulty: {results['difficulty']}")

# Option to Retrain (Checkbox example as requested)
if st.checkbox("Show Retrain Option"):
    if st.button("Train Model Now"):
        with st.spinner("Training..."):
            worker.train("final.csv")
            st.success("Training Done!")
