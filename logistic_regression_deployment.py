"""
Logistic Regression Deployment Script
-----------------------------------
This script handles the machine learning pipeline for classifying educational 
questions into Bloom's Taxonomy levels and Difficulty levels.

The pipeline includes:
1. Data Loading: Reading from CSV.
2. Preprocessing: Cleaning data and engineering features (success rates, text length, etc.).
3. Feature Extraction: 
    - Text: Sentence Transformers (MiniLM) for semantic embeddings.
    - Categorical: One-Hot Encoding for subjects and topics.
    - Numerical: Standard Scaling for quantitative metrics.
4. Model Management: Training, saving (joblib), and loading models for inference.
"""

import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from scipy.sparse import hstack



class BloomModelDeployer:
    """
    A robust class to manage the lifecycle of Bloom Level and Difficulty models.
    """
    
    def __init__(self, model_dir="models"):
        """
        Setup the model directory and define the core features we care about.
        """
        self.model_dir = model_dir
        self.bloom_model = None
        self.difficulty_model = None
        self.ohe = None
        self.scaler = None
        self.sentence_model = None
        
        # We track these specific numerical metrics to help the model decide
        # how hard a question is and what cognitive level it targets.
        self.numerical_cols = [
            "avg_score", "correct_percentage", "num_students_attempted",
            "num_students_correct", "time_taken_minutes", "success_rate",
            "log_attempts", "question_length"
        ]
        
        # Ensure we have a place to save our trained models
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def _get_sentence_transformer(self):
        """
        Loads the embedding model only when needed to save startup time and memory.
        We use 'all-MiniLM-L6-v2' because it strikes a great balance between speed and quality.
        We explicitly use 'cpu' to avoid "meta tensor" device errors on some systems.
        """
        if self.sentence_model is None:
            print("Loading text embedding model (this might take a few seconds)...")
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self.sentence_model

    def preprocess(self, df):
        """
        The cleaning engine. It takes raw table data and turns it into 
        meaningful features that the math inside the models can understand.
        """
        # Always work on a copy to prevent accidental changes to the original input
        data = df.copy()
        
        # Calculate 'success_rate': how often students get it right.
        # We use .replace(0, 1) to avoid that annoying "division by zero" error if no one tried it.
        data["success_rate"] = data["num_students_correct"] / data["num_students_attempted"].replace(0, 1)
        
        # 'log_attempts': we take the log because some questions have thousands of hits, 
        # and others have five. Logging "squashes" this range so it's easier to model.
        data["log_attempts"] = np.log1p(data["num_students_attempted"])
        
        # 'question_length': longer questions might imply higher complexity or cognitive load.
        data["question_length"] = data["question_text"].astype(str).apply(lambda x: len(x.split()))
        
        # Fill in any missing spots with the median (the "middle" value).
        # This keeps the model from crashing on empty data points.
        for col in self.numerical_cols:
            if col in data.columns:
                data[col] = data[col].fillna(data[col].median())
                
        return data

    def train(self, data_path="final.csv"):
        """
        The heavy lifter. It reads the CSV, prepares the features, trains two models,
        and saves everything so we can use it later without retraining.
        """
        if not os.path.exists(data_path):
            print(f"File not found: {data_path}")
            return

        print(f"--- Starting Training Pipeline using {data_path} ---")
        df = pd.read_csv(data_path)
        
        # Clean up any leftover index columns from the CSV
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns='Unnamed: 0')
            
        # Get rid of duplicate questions so the model doesn't just "memorize" them
        df = df.drop_duplicates(subset=['question_text'])
        df = self.preprocess(df)
        
        # Separate our input data (X) from what we want to predict (y)
        X = df.drop(columns=["bloom_level", "difficulty"])
        y_bloom = df["bloom_level"]
        y_difficulty = df["difficulty"]
        
        # Split the data so we can test the model on questions it has NEVER seen before.
        # 80% for learning, 20% for testing.
        X_train, X_test, yb_train, yb_test = train_test_split(X, y_bloom, test_size=0.2, random_state=42)
        _, _, yd_train, yd_test = train_test_split(X, y_difficulty, test_size=0.2, random_state=42)

        # 1. Logic for text: Convert questions into numbers (Embeddings)
        st = self._get_sentence_transformer()
        print("Transforming text into vectors...")
        X_train_text = st.encode(X_train["question_text"].astype(str).tolist(), show_progress_bar=True)
        X_test_text = st.encode(X_test["question_text"].astype(str).tolist(), show_progress_bar=True)
        
        # 2. Logic for categories: One-Hot Encoding (OHE)
        # This turns 'English' or 'Math' into columns of 0s and 1s.
        print("Encoding categories...")
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        X_train_cat = self.ohe.fit_transform(X_train[["subject", "topic"]])
        X_test_cat = self.ohe.transform(X_test[["subject", "topic"]])
        
        # 3. Logic for numbers: Scaling
        # This makes sure 'avg_score' (0-1) and 'students' (0-1000) have equal weight.
        print("Scaling numerical metrics...")
        self.scaler = StandardScaler()
        X_train_num_scaled = self.scaler.fit_transform(X_train[self.numerical_cols].values)
        X_test_num_scaled = self.scaler.transform(X_test[self.numerical_cols].values)
        
        # Combine all features (Text + Meta + Stats) into one big input matrix
        X_train_final = hstack([X_train_text, X_train_cat, X_train_num_scaled])
        X_test_final = hstack([X_test_text, X_test_cat, X_test_num_scaled])
        
        # --- Train the Bloom Model ---
        # We use 'balanced' weights because some levels (like 'Remember') are way more common than others.
        print("Fitting Bloom Level model...")
        self.bloom_model = LogisticRegression(max_iter=5000, class_weight="balanced")
        self.bloom_model.fit(X_train_final, yb_train)
        
        # --- Train the Difficulty Model ---
        print("Fitting Difficulty model...")
        self.difficulty_model = LogisticRegression(max_iter=5000)
        self.difficulty_model.fit(X_train_final, yd_train)
        
        # Save our work to disk!
        self._save_assets()
        
        # Show the user how well we did
        print("\n=== Model performance summary ===\n")
        print("Bloom Level Accuracy:")
        print(classification_report(yb_test, self.bloom_model.predict(X_test_final)))
        print("-" * 30)
        print("Difficulty Accuracy:")
        print(classification_report(yd_test, self.difficulty_model.predict(X_test_final)))

    def _save_assets(self):
        """Saves models and encoders so we don't have to train every time."""
        print(f"Saving models and transformers to the '{self.model_dir}' folder...")
        joblib.dump(self.bloom_model, os.path.join(self.model_dir, "bloom_model.pkl"))
        joblib.dump(self.difficulty_model, os.path.join(self.model_dir, "difficulty_model.pkl"))
        joblib.dump(self.ohe, os.path.join(self.model_dir, "ohe_encoder.pkl"))
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.pkl"))
        print("Saving complete.")

    def load_models(self):
        """Loads the saved models. Returns True if successful."""
        try:
            self.bloom_model = joblib.load(os.path.join(self.model_dir, "bloom_model.pkl"))
            self.difficulty_model = joblib.load(os.path.join(self.model_dir, "difficulty_model.pkl"))
            self.ohe = joblib.load(os.path.join(self.model_dir, "ohe_encoder.pkl"))
            self.scaler = joblib.load(os.path.join(self.model_dir, "scaler.pkl"))
            # Pre-load the transformer so prediction is snappy
            self._get_sentence_transformer()
            return True
        except Exception as e:
            print(f"Could not load models: {e}. Maybe you need to --train first?")
            return False

    def predict(self, question_data):
        """
        Takes a single dictionary of question features and predicts the outputs.
        """
        # Ensure models are loaded before we try to use them
        if not self.bloom_model:
            if not self.load_models():
                raise Exception("Models not available for prediction.")

        # Preprocess the single input point
        input_df = pd.DataFrame([question_data])
        processed_df = self.preprocess(input_df)
        
        # Get features ready for the model
        st = self._get_sentence_transformer()
        text_vec = st.encode(processed_df["question_text"].astype(str).tolist())
        cat_vec = self.ohe.transform(processed_df[["subject", "topic"]])
        num_vec = self.scaler.transform(processed_df[self.numerical_cols].values)
        
        # Merge everything
        final_x = hstack([text_vec, cat_vec, num_vec])
        
        # Get the results!
        return {
            "bloom_level": self.bloom_model.predict(final_x)[0],
            "difficulty": self.difficulty_model.predict(final_x)[0]
        }

if __name__ == "__main__":
    # Command line help and settings
    parser = argparse.ArgumentParser(description="Logistic Regression Deployer")
    parser.add_argument("--train", action="store_true", help="Train the models using final.csv")
    args = parser.parse_args()

    worker = BloomModelDeployer()

    if args.train:
        worker.train("final.csv")
    else:
        print("Run with --train to build your models, or use it as a library in your app!")
