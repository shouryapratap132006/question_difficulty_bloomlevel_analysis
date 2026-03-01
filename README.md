---
title: Question Level Prediction
emoji: 🎓
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
app_file: app.py
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

## ⚙️ Project Workflow

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

## 📈 Results

| Metric (Accuracy) | Bloom Level | Difficulty |
|---|---|---|
| **Accuracy Score** | **0.33** | **0.40** |
| **F1 Score (Macro)** | 0.33 | 0.39 |

> [!IMPORTANT]
> The current accuracy levels are primarily limited by the **synthesized nature of the dataset**. LLM-generated data, while useful for bootstrapping, often lacks the subtle nuances of real-world educational assessments, which affects the model's ability to reach higher precision.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **Python 3.9** | Programming language |
| **Streamlit** | UI Framework & Dashboard |
| **Sentence-Transformers** | NLP & Semantic Embeddings |
| **scikit-learn** | ML Models, Preprocessing, and Metrics |
| **Pandas / NumPy** | Data manipulation and Numerical logic |
| **Docker** | Containerization |
| **Hugging Face** | Deployment |

---

## 📁 Project Structure

```
capstone_genai/
├── models/                    # Saved .pkl joblib artifacts
├── final.csv                  # Main dataset
├── milestone1.ipynb           # Research and Development Notebook
├── logistic_regression_deployment.py # Core ML class
├── app.py                     # Beautified Streamlit Frontend
├── requirements.txt           # Deployment dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Pip (Python Package Manager)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train the models (if pkl files are missing)
python3 logistic_regression_deployment.py --train

# Launch the dashboard
streamlit run app.py
```

### Running with Docker

```bash
# Build the Docker image
docker build -t question-classifier .

# Run the container
docker run -p 7860:7860 question-classifier
```

---

## 📝 License

This project is part of the GenAI Capstone project.
