"""
generate_models.py — Train the Difficulty Classifier

Loads the StackOverflow-style questions dataset, cleans question text,
labels difficulty using score percentiles (same method as the app),
extracts TF-IDF features, trains a Logistic Regression model with
proper train/test split and 5-fold cross-validation, then saves the
model and metrics to the models/ directory.

Dataset: StackOverflow questions (https://www.kaggle.com/datasets/stackoverflow/stackoverflow)
Difficulty labeling: Score percentile based (bottom 33% = Hard, top 33% = Easy)

Usage:
    python generate_models.py
"""

import os
import re
import json
import html
import numpy as np
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "data/training_questions.csv"
MODEL_DIR = "models"
TEST_SIZE = 0.20          # 80/20 split
CV_FOLDS = 5              # 5-fold cross-validation
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 3000

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean raw question text for TF-IDF vectorization.
    Strips HTML, lowercases, removes non-alphabetic characters,
    and collapses whitespace.
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)   # strip HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Main Training Pipeline ───────────────────────────────────────────────────
def main():
    # 1. Load dataset
    print(f"Loading dataset from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH, encoding="latin1")
    print(f"  Loaded {len(df)} questions")

    # 2. Validate required columns
    assert "Title" in df.columns, "Dataset must have a 'Title' column"
    assert "Score" in df.columns, "Dataset must have a 'Score' column"

    # 3. Assign difficulty labels using score percentiles
    #    Same method used in app.py → Difficulty Analysis page
    #    Bottom 33% by score = Hard, top 33% = Easy, middle = Medium
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Score"])

    q_low = df["Score"].quantile(0.33)
    q_high = df["Score"].quantile(0.66)

    df["Difficulty"] = pd.cut(
        df["Score"],
        bins=[-np.inf, q_low, q_high, np.inf],
        labels=["Hard", "Medium", "Easy"],
    )
    print(f"  Score thresholds: Hard ≤ {q_low:.0f}, Medium ≤ {q_high:.0f}, Easy > {q_high:.0f}")

    # 4. Combine Title + Body for richer text features
    df["Body"] = df["Body"].fillna("")
    df["Title"] = df["Title"].fillna("")
    df["combined_text"] = df["Title"] + " " + df["Body"]

    # 5. Clean text
    print("Cleaning text ...")
    df["cleaned"] = df["combined_text"].apply(clean_text)

    # 6. Drop rows with empty text or missing labels
    df = df.dropna(subset=["Difficulty"])
    df = df[df["cleaned"].str.len() > 0]
    print(f"  {len(df)} questions after cleaning")

    # 7. Show class distribution
    print("\nClass distribution:")
    for label, count in df["Difficulty"].value_counts().items():
        print(f"  {label}: {count} ({count / len(df) * 100:.1f}%)")

    # 8. Prepare features and labels
    X_text = df["cleaned"].tolist()
    y = df["Difficulty"].astype(str).tolist()

    # 9. TF-IDF vectorization
    print(f"\nFitting TF-IDF vectorizer (max_features={MAX_TFIDF_FEATURES}) ...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        ngram_range=(1, 2),     # unigrams + bigrams
        stop_words="english",
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"  Feature matrix shape: {X_tfidf.shape}")

    # 10. Train/Test split (stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # 11. Cross-validation on training set
    print(f"\nRunning {CV_FOLDS}-fold cross-validation on training set ...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    print(f"  CV Accuracy per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy:     {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # 12. Train final model on full training set
    print("\nTraining final model ...")
    model.fit(X_train, y_train)

    # 13. Evaluate on held-out test set
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    print(f"\n── Test Set Results ──")
    print(f"  Accuracy: {test_acc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    print(f"Confusion Matrix:\n{cm}")

    # 14. Save model, vectorizer, and metrics
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_regression_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    np.save(os.path.join(MODEL_DIR, "confusion_matrix.npy"), cm)

    metrics = {
        "accuracy": test_acc,
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "cv_scores": cv_scores.tolist(),
        "report": report,
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "total_samples": len(df),
        "tfidf_features": int(X_tfidf.shape[1]),
        "score_thresholds": {"low": float(q_low), "high": float(q_high)},
    }
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Model and metrics saved to {MODEL_DIR}/")
    print(f"   logistic_regression_model.pkl")
    print(f"   tfidf_vectorizer.pkl")
    print(f"   confusion_matrix.npy")
    print(f"   model_metrics.json")


if __name__ == "__main__":
    main()
