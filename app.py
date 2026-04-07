import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os
import re
import html
import json
import sys
sys.path.append("src")

from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from agents.analyzer import analyze_difficulty

import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Exam Question Analysis System",
    page_icon="📝",
    layout="wide",
)

# ──────────────────────────────────────────────
# Custom CSS for a clean, modern look
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
    }
    .metric-card p {
        margin: 0;
        font-size: 0.9rem;
        opacity: 0.85;
    }
    .card {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Sidebar Navigation
# ──────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/exam.png", width=80)
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📤 Upload Data",
        "📊 Difficulty Analysis",
        "👨‍🎓 Student Performance",
        "📈 Visualizations",
        "🤖 Model Evaluation",
        "👩‍🏫 Assessment Assistant",
    ],
)

st.sidebar.markdown("---")
st.sidebar.info("**Milestone 1** – ML-Based Exam Question Analytics")

# ──────────────────────────────────────────────
# Session State for uploaded data
# ──────────────────────────────────────────────
if "questions_df" not in st.session_state:
    st.session_state.questions_df = None
if "responses_df" not in st.session_state:
    st.session_state.responses_df = None


# ══════════════════════════════════════════════
# Helper: clean text (same pipeline as notebook)
# ══════════════════════════════════════════════
def clean_text_pipeline(text):
    if pd.isna(text) or not isinstance(text, str):
        return ""
    text = html.unescape(text)
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
    except Exception:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = [w for w in text.split() if w not in stop_words]
    return " ".join(tokens).strip()


# ══════════════════════════════════════════════
# PAGE: Home
# ══════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown('<p class="main-header">📝 Intelligent Exam Question Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-driven system to assess exam question quality, difficulty, and student performance patterns.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📤</h3>
            <p>Upload exam questions & student responses</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
            <h3>📊</h3>
            <p>Analyze difficulty & quality classifications</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
            <h3>📈</h3>
            <p>Visualize performance trends & insights</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🔑 Key Features")
    features = {
        "Question Difficulty Prediction": "Classify questions as Easy, Medium, or Hard using Logistic Regression.",
        "Student Performance Patterns": "Analyze how students respond to each question.",
        "Text Feature Extraction": "TF-IDF based feature extraction from question text.",
        "Model Evaluation": "Accuracy, Precision, Recall & Confusion Matrix for Logistic Regression.",
        "Visual Insights": "Interactive charts for question-wise performance trends.",
    }
    for title, desc in features.items():
        st.markdown(f"""
        <div class="card">
            <strong>{title}</strong><br/>
            <span style="color:#666">{desc}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📐 System Architecture")
    st.markdown("""
    ```
    ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
    │  Raw Data     │───▶│  Preprocessing   │───▶│ Feature         │
    │  (CSV Upload) │    │  & Text Cleaning │    │ Extraction      │
    └──────────────┘    └──────────────────┘    │ (TF-IDF)        │
                                                 └────────┬────────┘
                                                          │
                         ┌──────────────────┐    ┌────────▼────────┐
                         │ Difficulty        │◀───│ Logistic        │
                         │ Classification    │    │ Regression      │
                         └────────┬─────────┘    └─────────────────┘
                                  │
                         ┌────────▼─────────┐
                         │  Dashboard &      │
                         │  Visualizations   │
                         └──────────────────┘
    ```
    """)

# ══════════════════════════════════════════════
# PAGE: Upload Data
# ══════════════════════════════════════════════
elif page == "📤 Upload Data":
    st.markdown('<p class="main-header">📤 Upload Data</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload your exam questions and student response data (CSV format).</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Exam Questions")
        st.caption("Expected columns: `Id`, `Title`, `Body`, `Score`, `Tags` (or similar)")
        questions_file = st.file_uploader("Upload Questions CSV", type=["csv"], key="q_upload")
        if questions_file is not None:
            st.session_state.questions_df = pd.read_csv(questions_file, encoding="latin1", nrows=5000)
            st.success(f"✅ Loaded {len(st.session_state.questions_df)} questions!")
            st.dataframe(st.session_state.questions_df.head(10), use_container_width=True)

    with col2:
        st.subheader("👨‍🎓 Student Responses / Answers")
        st.caption("Expected columns: `Id`, `ParentId`, `Body`, `Score` (or similar)")
        responses_file = st.file_uploader("Upload Responses CSV", type=["csv"], key="r_upload")
        if responses_file is not None:
            st.session_state.responses_df = pd.read_csv(responses_file, encoding="latin1", nrows=5000)
            st.success(f"✅ Loaded {len(st.session_state.responses_df)} responses!")
            st.dataframe(st.session_state.responses_df.head(10), use_container_width=True)

    st.markdown("---")
    if st.session_state.questions_df is not None:
        st.subheader("📋 Questions Data Summary")
        st.write(st.session_state.questions_df.describe())
    if st.session_state.responses_df is not None:
        st.subheader("📋 Responses Data Summary")
        st.write(st.session_state.responses_df.describe())


# ══════════════════════════════════════════════
# PAGE: Difficulty Analysis
# ══════════════════════════════════════════════
elif page == "📊 Difficulty Analysis":
    st.markdown('<p class="main-header">📊 Difficulty Classification</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Questions classified as Easy, Medium, or Hard based on their score distribution.</p>', unsafe_allow_html=True)

    if st.session_state.questions_df is not None:
        df = st.session_state.questions_df.copy()

        # Simple difficulty classification based on Score column
        score_col = None
        for col in ["Score", "score", "OwnerUserId"]:
            if col in df.columns:
                score_col = col
                break

        if score_col and pd.api.types.is_numeric_dtype(df[score_col]):
            q_low = df[score_col].quantile(0.33)
            q_high = df[score_col].quantile(0.66)

            df["Difficulty"] = pd.cut(
                df[score_col],
                bins=[-np.inf, q_low, q_high, np.inf],
                labels=["Hard", "Medium", "Easy"],
            )

            # Build and store structured difficulty distribution for Milestone 2 agents
            easy_count  = int((df["Difficulty"] == "Easy").sum())
            medium_count = int((df["Difficulty"] == "Medium").sum())
            hard_count  = int((df["Difficulty"] == "Hard").sum())

            difficulty_distribution = {
                "Easy": easy_count,
                "Medium": medium_count,
                "Hard": hard_count,
                "total": len(df),
                "thresholds": {"low": float(q_low), "high": float(q_high)},
                "percentages": {
                    "Easy":   round(easy_count   / len(df) * 100, 1),
                    "Medium": round(medium_count  / len(df) * 100, 1),
                    "Hard":   round(hard_count    / len(df) * 100, 1),
                },
            }
            st.session_state.difficulty_distribution = difficulty_distribution
            problems = analyze_difficulty(difficulty_distribution)
            st.session_state.analysis_problems = problems

            # Show metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", len(df))
            with col2:
                st.metric("🟢 Easy", easy_count)
            with col3:
                st.metric("🟡 Medium", medium_count)
            with col4:
                st.metric("🔴 Hard", hard_count)

            st.markdown("---")

            # Difficulty distribution bar chart
            st.subheader("Difficulty Distribution")
            diff_counts = df["Difficulty"].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            diff_counts.reindex(["Easy", "Medium", "Hard"]).plot(
                kind="bar", ax=ax, color=colors, edgecolor="white", linewidth=1.5
            )
            ax.set_ylabel("Number of Questions")
            ax.set_xlabel("Difficulty Level")
            ax.set_title("Question Difficulty Distribution")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.xticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("---")

            # Export difficulty distribution as JSON (consumed by Milestone 2 agents)
            st.subheader("📤 Export for Assessment Agent")
            st.download_button(
                label="⬇️ Download Difficulty Distribution (JSON)",
                data=json.dumps(difficulty_distribution, indent=2),
                file_name="difficulty_distribution.json",
                mime="application/json",
                help="This JSON is used as input for the Milestone 2 agentic assessment assistant.",
            )

            st.markdown("---")

            # Show the classified table
            st.subheader("Classified Questions")
            display_cols = [c for c in ["Id", "Title", "Score", score_col, "Difficulty"] if c in df.columns]
            display_cols = list(dict.fromkeys(display_cols))  # remove duplicates
            st.dataframe(df[display_cols].head(50), use_container_width=True)

        else:
            st.warning("⚠️ Could not find a numeric 'Score' column to classify difficulty. Please check your data.")
    else:
        st.info("👈 Please upload questions data first from the **Upload Data** page.")


# ══════════════════════════════════════════════
# PAGE: Student Performance
# ══════════════════════════════════════════════
elif page == "👨‍🎓 Student Performance":
    st.markdown('<p class="main-header">👨‍🎓 Student Performance Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze how students respond to questions and identify performance patterns.</p>', unsafe_allow_html=True)

    if st.session_state.responses_df is not None:
        df = st.session_state.responses_df.copy()

        st.subheader("📋 Response Data Overview")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Responses", len(df))
        with col2:
            if "Score" in df.columns:
                st.metric("Average Score", f"{df['Score'].mean():.2f}")

        st.markdown("---")

        # Score distribution
        if "Score" in df.columns:
            st.subheader("📊 Response Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            df["Score"].clip(-10, 50).hist(bins=30, ax=ax, color="#667eea", edgecolor="white")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Response Scores")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Per-question performance
        if "ParentId" in df.columns and "Score" in df.columns:
            st.markdown("---")
            st.subheader("📈 Per-Question Response Statistics")
            per_q = df.groupby("ParentId")["Score"].agg(["mean", "count", "std"]).reset_index()
            per_q.columns = ["Question ID", "Avg Score", "Response Count", "Score Std Dev"]
            per_q = per_q.sort_values("Response Count", ascending=False)
            st.dataframe(per_q.head(50), use_container_width=True)

            # Top and bottom performing questions
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🏆 Top Scoring Questions")
                st.dataframe(
                    per_q.nlargest(10, "Avg Score")[["Question ID", "Avg Score", "Response Count"]],
                    use_container_width=True,
                )
            with col2:
                st.subheader("⚠️ Lowest Scoring Questions")
                st.dataframe(
                    per_q.nsmallest(10, "Avg Score")[["Question ID", "Avg Score", "Response Count"]],
                    use_container_width=True,
                )

    else:
        st.info("👈 Please upload response/answer data first from the **Upload Data** page.")


# ══════════════════════════════════════════════
# PAGE: Visualizations
# ══════════════════════════════════════════════
elif page == "📈 Visualizations":
    st.markdown('<p class="main-header">📈 Visualizations & Trends</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Interactive charts to explore question and performance trends.</p>', unsafe_allow_html=True)

    if st.session_state.questions_df is not None:
        df = st.session_state.questions_df.copy()

        # Score distribution
        if "Score" in df.columns:
            st.subheader("📊 Question Score Distribution")
            fig, ax = plt.subplots(figsize=(8, 4))
            df["Score"].clip(-5, 100).hist(bins=40, ax=ax, color="#764ba2", edgecolor="white")
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Question Scores")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Questions over time
        date_col = None
        for c in ["CreationDate", "creation_date"]:
            if c in df.columns:
                date_col = c
                break

        if date_col:
            st.markdown("---")
            st.subheader("📅 Questions Over Time")
            df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
            df["_month"] = df["_date"].dt.to_period("M")
            monthly = df.groupby("_month").size()
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly.plot(kind="line", ax=ax, color="#f5576c", linewidth=2)
            ax.set_xlabel("Month")
            ax.set_ylabel("Number of Questions")
            ax.set_title("Questions Posted Over Time")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        # Score boxplot by difficulty
        if "Score" in df.columns:
            st.markdown("---")
            st.subheader("📦 Score by Difficulty Category")
            q_low = df["Score"].quantile(0.33)
            q_high = df["Score"].quantile(0.66)
            df["Difficulty"] = pd.cut(
                df["Score"],
                bins=[-np.inf, q_low, q_high, np.inf],
                labels=["Hard", "Medium", "Easy"],
            )
            fig, ax = plt.subplots(figsize=(6, 4))
            df.boxplot(column="Score", by="Difficulty", ax=ax)
            ax.set_title("Score Distribution by Difficulty")
            ax.set_xlabel("Difficulty")
            ax.set_ylabel("Score")
            plt.suptitle("")
            plt.tight_layout()
            st.pyplot(fig)

    else:
        st.info("👈 Please upload questions data first from the **Upload Data** page.")




# ══════════════════════════════════════════════
# PAGE: Model Evaluation
# ══════════════════════════════════════════════
elif page == "🤖 Model Evaluation":
    st.markdown('<p class="main-header">🤖 Model Evaluation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Summary of ML model performance for question difficulty prediction.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <strong>Model Used</strong><br/>
        <span style="color:#666">Logistic Regression classifier trained on TF-IDF features extracted from question text.</span>
    </div>
    """, unsafe_allow_html=True)

    # Try loading saved model
    MODEL_PATH = "models/logistic_regression_model.pkl"
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        tfidf_vec = joblib.load(VECTORIZER_PATH)

        st.success("✅ Trained model loaded successfully!")

        st.markdown("---")

        # ── Try It: Predict Difficulty ──
        st.subheader("🔮 Try It — Predict Question Difficulty")
        user_question = st.text_area(
            "Enter a question to predict its difficulty:",
            placeholder="e.g. How do I reverse a linked list in Python?",
            height=100,
        )

        if st.button("Predict Difficulty", type="primary"):
            if user_question.strip():
                cleaned = clean_text_pipeline(user_question)
                q_tfidf = tfidf_vec.transform([cleaned])
                prediction = model.predict(q_tfidf)[0]
                probabilities = model.predict_proba(q_tfidf)[0]

                # Color map for difficulty
                color_map = {"Easy": "#2ecc71", "Medium": "#f39c12", "Hard": "#e74c3c"}
                emoji_map = {"Easy": "🟢", "Medium": "🟡", "Hard": "🔴"}

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color_map.get(prediction, '#667eea')}22, {color_map.get(prediction, '#667eea')}44);
                    padding: 1.5rem; border-radius: 12px; border-left: 5px solid {color_map.get(prediction, '#667eea')};
                    margin: 1rem 0;">
                    <h3 style="margin:0">{emoji_map.get(prediction, '')} Predicted Difficulty: {prediction}</h3>
                </div>
                """, unsafe_allow_html=True)

                # Show confidence per class
                prob_df = pd.DataFrame({
                    "Difficulty": model.classes_,
                    "Confidence": [f"{p:.1%}" for p in probabilities],
                })
                st.table(prob_df)
            else:
                st.warning("Please enter a question first.")

        st.markdown("---")

        # ── Model Info ──
        st.subheader("📊 Model Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", "Logistic Regression")
        with col2:
            st.metric("Features", f"{len(tfidf_vec.get_feature_names_out())} TF-IDF")
        with col3:
            st.metric("Classes", ", ".join(model.classes_))

        st.markdown("---")
        st.info("💡 Upload your data on the **Upload Data** page to see difficulty analysis and student performance insights.")

        # ── Performance Metrics ──
        st.subheader("📈 Model Performance Metrics")
 
        METRICS_PATH = "models/model_metrics.json"
        CM_PATH = "models/confusion_matrix.npy"
 
        metrics_data = None
        cm_matrix = None
        cm_labels = list(model.classes_)
        metrics_source = None
 
        # 1) Try pre-saved metrics from notebook training
        if os.path.exists(METRICS_PATH):
            with open(METRICS_PATH) as f:
                metrics_data = json.load(f)
            if os.path.exists(CM_PATH):
                cm_matrix = np.load(CM_PATH)
            metrics_source = "Training test split (saved from notebook)"
 
        # 2) Fall back to computing live on uploaded questions data
        elif st.session_state.questions_df is not None:
            df_eval = st.session_state.questions_df.copy()
            has_score = "Score" in df_eval.columns
            has_text  = "Title" in df_eval.columns or "Body" in df_eval.columns
 
            if has_score and has_text:
                with st.spinner("Computing metrics on uploaded data…"):
                    q_lo = df_eval["Score"].quantile(0.33)
                    q_hi = df_eval["Score"].quantile(0.66)
                    df_eval["_label"] = pd.cut(
                        df_eval["Score"],
                        bins=[-np.inf, q_lo, q_hi, np.inf],
                        labels=["Hard", "Medium", "Easy"],
                    )
                    df_eval = df_eval.dropna(subset=["_label"])
 
                    title = df_eval["Title"].fillna("") if "Title" in df_eval.columns else pd.Series([""] * len(df_eval))
                    body  = df_eval["Body"].fillna("")  if "Body"  in df_eval.columns else pd.Series([""] * len(df_eval))
                    df_eval["_text"]    = title + " " + body
                    df_eval["_cleaned"] = df_eval["_text"].apply(clean_text_pipeline)
 
                    X_eval  = tfidf_vec.transform(df_eval["_cleaned"])
                    y_true  = df_eval["_label"].astype(str)
                    y_pred  = model.predict(X_eval)
 
                    acc    = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    cm_matrix = confusion_matrix(y_true, y_pred, labels=cm_labels)
 
                    metrics_data   = {"accuracy": acc, "report": report}
                    metrics_source = "Uploaded questions data (live evaluation)"
 
        if metrics_data is not None:
            st.caption(f"Source: {metrics_source}")
            acc    = metrics_data.get("accuracy", 0)
            report = metrics_data.get("report", {})
            macro  = report.get("macro avg", {})
 
            # Top-level metric cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Accuracy",           f"{acc:.1%}")
            with m2:
                st.metric("Precision (macro)",  f"{macro.get('precision', 0):.1%}")
            with m3:
                st.metric("Recall (macro)",     f"{macro.get('recall', 0):.1%}")
            with m4:
                st.metric("F1 Score (macro)",   f"{macro.get('f1-score', 0):.1%}")
 
            st.markdown("---")
 
            # Per-class breakdown table
            st.markdown("**Per-Class Metrics**")
            per_class_rows = []
            for cls in ["Easy", "Medium", "Hard"]:
                if cls in report:
                    per_class_rows.append({
                        "Class":     cls,
                        "Precision": f"{report[cls]['precision']:.3f}",
                        "Recall":    f"{report[cls]['recall']:.3f}",
                        "F1-Score":  f"{report[cls]['f1-score']:.3f}",
                        "Support":   int(report[cls]["support"]),
                    })
            if per_class_rows:
                st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)
 
            # Confusion matrix heatmap
            if cm_matrix is not None:
                st.markdown("---")
                st.markdown("**Confusion Matrix**")
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                im = ax_cm.imshow(cm_matrix, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, ax=ax_cm)
                ax_cm.set_xticks(range(len(cm_labels)))
                ax_cm.set_yticks(range(len(cm_labels)))
                ax_cm.set_xticklabels(cm_labels, fontsize=10)
                ax_cm.set_yticklabels(cm_labels, fontsize=10)
                ax_cm.set_xlabel("Predicted Label", fontsize=11)
                ax_cm.set_ylabel("True Label", fontsize=11)
                ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
                thresh = cm_matrix.max() / 2.0
                for i in range(cm_matrix.shape[0]):
                    for j in range(cm_matrix.shape[1]):
                        ax_cm.text(
                            j, i, str(cm_matrix[i, j]),
                            ha="center", va="center", fontsize=12, fontweight="bold",
                            color="white" if cm_matrix[i, j] > thresh else "black",
                        )
                plt.tight_layout()
                st.pyplot(fig_cm)
        else:
            st.info("💡 Upload questions data with a **Score** column on the Upload Data page to compute live model metrics, or run the notebook to save pre-computed metrics.")

    else:
        st.warning("⚠️ No saved model found. Please train and save the model from the notebook first.")
        st.caption(f"Expected model at: `{MODEL_PATH}` and vectorizer at: `{VECTORIZER_PATH}`")


# ══════════════════════════════════════════════
# PAGE: Assessment Assistant
# ══════════════════════════════════════════════
elif page == "👩‍🏫 Assessment Assistant":
    st.markdown('<p class="main-header">👩‍🏫 Assessment Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered advisor to assess exam difficulty and provide improvement recommendations.</p>', unsafe_allow_html=True)

    st.info("💡 **Milestone 2 Agentic Workflow**")
    
    if st.button("Run AI Analysis", type="primary"):
        import time
        
        # Simulate LangGraph Dummy Steps
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.text("Analyzing difficulty distribution... (Agent 1: Analyzer)")
        time.sleep(1.0)
        progress_bar.progress(25)
        
        status_text.text("Retrieving teaching principles... (Agent 2: Retriever)")
        time.sleep(1.0)
        progress_bar.progress(50)
        
        status_text.text("Generating recommendations... (Agent 3: Recommender)")
        time.sleep(1.0)
        progress_bar.progress(75)
        
        status_text.text("Formatting structured report... (Agent 4: Reporter)")
        time.sleep(1.0)
        progress_bar.progress(100)
        status_text.success("Analysis Complete!")
        
        try:
            from src.agents.reporter import generate_report
            from src.utils.pdf_export import create_pdf_report
            
            dummy_state = {
                "difficulty": {"Easy": "15%", "Medium": "25%", "Hard": "60%"},
                "problems": [
                    "60% of questions are Hard, indicating the exam may be too difficult.",
                    "Low average score across hard questions suggests students are struggling with complex reasoning.",
                    "Lack of easy questions may reduce student confidence early in the exam."
                ],
                "principles": [
                    "A well-balanced exam should ideally have 30% Easy, 40% Medium, and 30% Hard questions.",
                    "Assessments should begin with easier questions to build confidence.",
                    "Bloom's Taxonomy suggests evaluating a mix of recall, understanding, and application."
                ],
                "recommendations": [
                    "Replace 3-5 Hard questions with Easy recall questions on foundational concepts.",
                    "Ensure that complex questions are broken down into smaller, progressive sub-questions.",
                    "Review the hardest questions to ensure they are testing intended concepts, not just reading comprehension."
                ]
            }
            
            report_text = generate_report(dummy_state)
            pdf_bytes = create_pdf_report(report_text)
            
            st.markdown("---")
            st.subheader("📄 Generated Assessment Report")
            
            with st.expander("View Full Report", expanded=True):
                st.markdown(report_text)
                
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="Assessment_Report.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error executing report generation: {str(e)}")


st.sidebar.markdown("---")
st.sidebar.caption("Intelligent Exam Question Analysis System")

st.markdown("---")
st.subheader("AI Analysis (Agent 1)")

if "analysis_problems" in st.session_state:
    for p in st.session_state.analysis_problems:
        st.warning(f"{p}")
