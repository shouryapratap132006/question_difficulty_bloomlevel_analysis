---
title: Question Level Prediction
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
app_file: app.py
python_version: "3.9"
pinned: false
---

# 🎓 Question Bloom Level & Difficulty Prediction

A machine learning project that classifies educational questions into **Bloom's Taxonomy levels** and **Difficulty categories** using **Logistic Regression**. The project implements a custom NLP pipeline with **Sentence Transformers** and provides a beautified **Streamlit** interface for real-time predictions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

---

## 🔍 Overview

This project performs an end-to-end machine learning pipeline for analyzing the cognitive complexity and difficulty of educational questions. It covers:

1. **Data Exploration** — Analyzing question text distributions and student performance metadata
2. **Feature Engineering** — Generating semantic embeddings and normalizing student success metrics
3. **Model Training & Evaluation** — Developing calibrated Logistic Regression models for multi-class classification
4. **Deployment** — Creating a modern, interactive dashboard for production-ready inference

---

## 📊 Dataset

> [!NOTE]
> This dataset was synthesized using a Large Language Model (LLM) due to the scarcity of publicly available datasets for multi-class Bloom's Taxonomy classification on specific educational content.

| Property | Details |
|---|---|
| **File** | `final.csv` |
| **Rows** | 5,500 |
| **Columns** | ~11 predictive columns |
| **Target Variables** | `bloom_level`, `difficulty` |

### Features

The model uses a total of **12 features** (9 base features from the dataset + 3 engineered features):

| Feature | Type | Source | Description |
|---|---|---|---|
| **Question Text** | string | Base | The raw text of the question (Vectorized via NLP) |
| **Subject** | object | Base | Broad subject category (e.g., Science, Maths) |
| **Topic** | object | Base | Specific topic within the subject |
| **Avg Score** | float64 | Base | Average score achieved by students (0.0 - 1.0) |
| **Correct %** | float64 | Base | Percentage of students who got the question right |
| **Students Attempted** | int64 | Base | Total count of students who answered the question |
| **Students Correct** | int64 | Base | Total count of students who answered correctly |
| **Time Taken** | float64 | Base | Average time spent on the question (minutes) |
| **Success Rate** | float64 | **Engineered** | Calculated as (Correct / Attempted) |
| **Log Attempts** | float64 | **Engineered** | Log-transformed attempt count for better scaling |
| **Question Length** | int64 | **Engineered** | Total word count of the question text |

---

## ⚙️ Milestone 1: ML Predictive Modeling

### 1. Data Cleaning & Engineering
- **NLP Processing**: Text is vectorized using `SentenceTransformer` ('all-MiniLM-L6-v2') to capture semantic intent.
- **Categorical Encoding**: One-Hot Encoding applied to Subject and Topic features.
- **Scaling**: Standardized numerical metrics using `StandardScaler` for model stability.

### 2. Implementation Approach
- **Standalone Module**: All logic encapsulated in `BloomModelDeployer` class for modular usage.
- **Balanced Weights**: Implemented `class_weight='balanced'` to handle imbalanced levels in Bloom's Taxonomy.

### 3. Training & Evaluation
- Splitting data into 80% training and 20% testing sets.
- **Model Selection**: During experimentation, **XGBoost** and **Random Forest** were tested. However, they did not provide a significant improvement in accuracy for this specific categorical text task, leading to the selection of **Logistic Regression** for its better generalization and interpretability.
- Persistence of all artifacts (models, encoders, scalers) into the `models/` directory.

---

## 📈 Milestone 1: Results

| Metric (Accuracy) | Bloom Level | Difficulty |
|---|---|---|
| **Accuracy Score** | **0.33** | **0.40** |
| **F1 Score (Macro)** | 0.33 | 0.39 |

> [!IMPORTANT]
> The current accuracy levels are primarily limited by the **synthesized nature of the dataset**. LLM-generated data, while useful for bootstrapping, often lacks the subtle nuances of real-world educational assessments, which affects the model's ability to reach higher precision.

---

| Library | Purpose |
|---|---|
| **Python 3.9** | Programming language |
| **Streamlit** | UI Framework & Dashboard |
| **Sentence-Transformers** | NLP & Semantic Embeddings for ML/RAG |
| **scikit-learn** | ML Models, Preprocessing, and Metrics |
| **LangGraph / LangChain** | Multi-Agent Orchestration & Core AI logic |
| **FAISS** | In-Memory Vector Database |
| **Groq (Llama 3.1)** | Free-tier, ultra-fast LLM API |
| **Pandas / NumPy** | Data manipulation and Numerical logic |
| **Docker** | Containerization |

---

## 📁 Project Structure

```
capstone_genai/
├── data/
│   ├── final.csv                  # Main dataset
│   └── pedagogy_guidelines.md     # RAG document source
├── agent/                         # Multi-Agent LangGraph Logic
│   ├── graph.py                   # Agent workflow compilation
│   ├── nodes.py                   # Sub-agent logical nodes
│   ├── rag.py                     # FAISS Vector Store logic
│   └── state.py                   # Shared typed dictionary 
├── notebooks/
│   ├── milestone1.ipynb           # Model Research and Training
│   ├── milestone2.ipynb           # Agentic Assistant Execution
│   └── rag.ipynb                  # Vector Search Isolation Testing
├── models/                        # Saved Joblib Pickles
├── app.py                         # Beautiful Streamlit Dashboard
├── logistic_regression_deployment.py # ML Deployment Module
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🤖 Milestone 2: Agentic AI Assistant

The second milestone extends the ML foundational model by introducing a multi-agent orchestrated workflow using **LangGraph** paired with **Retrieval-Augmented Generation (RAG)** to provide highly specific instructional recommendations.

### 1. Knowledge Base & Vector Store (RAG)
- **Pedagogical Corpus**: We constructed `pedagogy_guidelines.md` consisting of structured Bloom's Taxonomy definitions, learning gap heuristics, and actionable question-refinement strategies.
- **Embedding & Storage**: The corpus is split into chunks and embedded locally using the identical `sentence-transformers/all-MiniLM-L6-v2` model from Milestone 1 for maximum efficiency. It is stored in memory using a **FAISS** vector database for split-second contextual retrieval.

### 2. Multi-Agent Workflow (LangGraph)
The LLM inference is rigorously structured via an explicit **StateGraph** architecture:
- **QuestionAnalyzer Node**: Evaluates the initial question's properties and imports the Logistic Regression model predictions.
- **GapDetector Node**: Calculates learning gaps algorithmically based on success rates and completion time.
- **RAGRetriever Node**: Executes a semantic similarity search against the FAISS vector store to retrieve appropriate pedagogical guidelines mapped to the detected gaps.
- **RecommendationGenerator Node**: Constructs a structured prompt encompassing the stats, gaps, and RAG context, sending it to the **Groq LLM (Llama 3.1)** API to generate natural, actionable redesign efforts.
- **ReportBuilder Node**: Formats all state artifacts into a unified dictionary for frictionless dashboard rendering.

### 3. Integrated UI Dashboard
- Transformed the legacy form into a unified, seamless interface leveraging native container components (`st.container`, `st.metric`).
- ML classifications and Agentic Recommendations run chronologically on a single button press.
- LLM feedback is formatted inside `st.chat_message` components for an intuitive "AI Co-pilot" collaborative experience.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Groq API Key

### Installation

```bash
# Create an environments file and add your Groq API key
echo "GROQ_API_KEY=your_key_here" > .env

# Install dependencies
python3 -m pip install -r requirements.txt

# Train the models (if pkl files are missing)
python3 logistic_regression_deployment.py --train

# Launch the dashboard
streamlit run app.py
```

### Running with Docker

```bash
# Build the Docker image
docker build -t question-classifier .

# Run the container (Make sure .env contains API keys)
docker run -p 7860:7860 --env-file .env question-classifier
```

---

## 📝 License

This project is part of the GenAI Capstone project.
