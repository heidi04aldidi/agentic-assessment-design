import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
