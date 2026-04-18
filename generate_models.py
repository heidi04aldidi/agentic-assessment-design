"""
generate_models.py — Train the Difficulty Classifier

Loads the StackOverflow stacksample dataset (Kaggle), processes questions
and answers, labels difficulty using Bayesian-smoothed answer scores,
extracts TF-IDF features, trains a Logistic Regression model with
proper train/test split and 5-fold cross-validation.

Dataset: https://www.kaggle.com/datasets/stackoverflow/stacksample
Setup:  Place Questions.csv, Answers.csv, Tags.csv in data/raw/

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
RAW_DIR = "data/raw"
MODEL_DIR = "models"
SAMPLE_FRAC = 0.4          # stratified sample fraction
TEST_SIZE = 0.20           # 80/20 split
CV_FOLDS = 5               # 5-fold cross-validation
RANDOM_STATE = 42
MAX_TFIDF_FEATURES = 10000

os.makedirs(MODEL_DIR, exist_ok=True)


# ── Text Cleaning ────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean raw question/answer text for ML processing.
    Strips HTML tags, lowercases, removes non-alphabetic characters,
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
    questions_path = os.path.join(RAW_DIR, "Questions.csv")
    answers_path = os.path.join(RAW_DIR, "Answers.csv")

    # ── 1. Load raw data ─────────────────────────────────────────────────
    print("Loading StackOverflow Questions ...")
    questions_df = pd.read_csv(questions_path, encoding="latin1")
    print(f"  Questions: {len(questions_df)} rows")

    print("Loading StackOverflow Answers ...")
    answers_df = pd.read_csv(answers_path, encoding="latin1")
    print(f"  Answers:   {len(answers_df)} rows")

    # ── 2. Clean question text ───────────────────────────────────────────
    print("\nCleaning question text ...")
    questions_df["clean_text"] = (
        questions_df["Title"].fillna("") + " " + questions_df["Body"].fillna("")
    ).apply(clean_text)
    questions_df = questions_df[questions_df["clean_text"].str.len() > 0]
    print(f"  {len(questions_df)} questions after cleaning")

    # ── 3. Clean answer text & compute answer stats ──────────────────────
    print("Computing answer statistics ...")
    answers_df["clean_text"] = answers_df["Body"].fillna("").apply(clean_text)

    answer_stats = answers_df.groupby("ParentId").agg(
        avg_answer_score=("Score", "mean"),
        answer_count=("Score", "count"),
        max_answer_score=("Score", "max"),
    ).reset_index()
    answer_stats.rename(columns={"ParentId": "question_id"}, inplace=True)
    print(f"  Answer stats for {len(answer_stats)} questions")

    # ── 4. Bayesian smoothing + normalize scores ─────────────────────────
    global_mean = answer_stats["avg_answer_score"].mean()
    C = answer_stats["answer_count"].mean()

    answer_stats["bayesian_avg_score"] = (
        (C * global_mean + answer_stats["answer_count"] * answer_stats["avg_answer_score"])
        / (C + answer_stats["answer_count"])
    )

    min_score = answer_stats["bayesian_avg_score"].min()
    max_score = answer_stats["bayesian_avg_score"].max()
    answer_stats["avg_score_normalized"] = (
        (answer_stats["bayesian_avg_score"] - min_score) / (max_score - min_score)
    )

    # ── 5. Assign difficulty labels using thresholds ─────────────────────
    def assign_difficulty(score):
        if score >= 0.035:
            return "Easy"
        elif score >= 0.020:
            return "Medium"
        else:
            return "Hard"

    answer_stats["difficulty"] = answer_stats["avg_score_normalized"].apply(assign_difficulty)

    print("\nRaw difficulty distribution:")
    for label, count in answer_stats["difficulty"].value_counts().items():
        print(f"  {label}: {count}")

    # ── 6. Merge questions with difficulty labels ────────────────────────
    questions_with_labels = questions_df.merge(
        answer_stats[["question_id", "difficulty"]],
        left_on="Id",
        right_on="question_id",
        how="inner",
    )
    print(f"\n  {len(questions_with_labels)} questions after merge with answer stats")

    # ── 7. Stratified sample (to keep training manageable) ───────────────
    questions_sampled = (
        questions_with_labels
        .groupby("difficulty", group_keys=False)
        .apply(lambda grp: grp.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE),
               include_groups=False)
        .reset_index(drop=True)
    )

    # Re-attach difficulty from the index if dropped by apply
    if "difficulty" not in questions_sampled.columns:
        # Fallback: simple stratified sample
        questions_sampled = questions_with_labels.sample(
            frac=SAMPLE_FRAC, random_state=RANDOM_STATE
        ).reset_index(drop=True)
    print(f"  Sampled {len(questions_sampled)} questions ({SAMPLE_FRAC*100:.0f}% stratified)")

    print("\nSampled class distribution:")
    for label, count in questions_sampled["difficulty"].value_counts().items():
        print(f"  {label}: {count} ({count / len(questions_sampled) * 100:.1f}%)")

    # ── 8. Prepare features and labels ───────────────────────────────────
    X_text = questions_sampled["clean_text"].fillna("").tolist()
    y = questions_sampled["difficulty"].tolist()

    # ── 9. TF-IDF vectorization ──────────────────────────────────────────
    print(f"\nFitting TF-IDF vectorizer (max_features={MAX_TFIDF_FEATURES}) ...")
    vectorizer = TfidfVectorizer(
        max_features=MAX_TFIDF_FEATURES,
        min_df=5,
        max_df=0.8,
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X_tfidf = vectorizer.fit_transform(X_text)
    print(f"  Feature matrix shape: {X_tfidf.shape}")

    # ── 10. Train/Test split ─────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"\n  Train set: {X_train.shape[0]} samples")
    print(f"  Test set:  {X_test.shape[0]} samples")

    # ── 11. Cross-validation on training set ─────────────────────────────
    print(f"\nRunning {CV_FOLDS}-fold cross-validation on training set ...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=3000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    cv_scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="accuracy")
    print(f"  CV Accuracy per fold: {[f'{s:.3f}' for s in cv_scores]}")
    print(f"  CV Mean Accuracy:     {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # ── 12. Train final model on full training set ───────────────────────
    print("\nTraining final Logistic Regression model ...")
    model.fit(X_train, y_train)

    # ── 13. Evaluate on held-out test set ────────────────────────────────
    y_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    print(f"\n{'═' * 40}")
    print(f"  Test Set Accuracy: {test_acc:.3f}")
    print(f"{'═' * 40}")
    print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")
    print(f"Confusion Matrix:\n{cm}")

    # ── 14. Feature importance (top words per class) ─────────────────────
    feature_names = vectorizer.get_feature_names_out()
    print("\nTop 10 TF-IDF features per class:")
    for i, class_label in enumerate(model.classes_):
        top10_idx = np.argsort(model.coef_[i])[-10:]
        top10_words = [feature_names[j] for j in top10_idx]
        print(f"  {class_label}: {top10_words}")

    # ── 15. Save model, vectorizer, and metrics ──────────────────────────
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
        "total_samples": len(questions_sampled),
        "tfidf_features": int(X_tfidf.shape[1]),
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
