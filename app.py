import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
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

st.set_page_config(
    page_title="ExamIQ — Exam Question Analysis",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>📋</text></svg>",
    layout="wide",
)

# ─── Global CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

    /* ── Reset & Base ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: #F7F6F3;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #1C1C1E !important;
        border-right: 1px solid #2C2C2E;
    }
    section[data-testid="stSidebar"] * {
        color: #E5E5E7 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 0.85rem !important;
        letter-spacing: 0.02em;
        padding: 6px 0;
        color: #AEAEB2 !important;
        transition: color 0.15s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #2C2C2E !important;
        opacity: 1 !important;
    }
    section[data-testid="stSidebar"] .stSuccess,
    section[data-testid="stSidebar"] .stInfo {
        background: #2C2C2E !important;
        border: 1px solid #3C3C3E !important;
        color: #E5E5E7 !important;
        border-radius: 8px;
        font-size: 0.78rem;
    }

    /* ── Main Content ── */
    .main .block-container {
        padding: 2.5rem 2.5rem 3rem;
        max-width: 1200px;
    }

    /* ── Page Header ── */
    .page-eyebrow {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6B6B6B;
        margin-bottom: 4px;
    }
    .page-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1C1C1E;
        line-height: 1.2;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #6B6B6B;
        margin-bottom: 2rem;
        font-weight: 300;
        line-height: 1.5;
    }
    .page-divider {
        height: 1px;
        background: linear-gradient(90deg, #E5E5EA 0%, transparent 100%);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Stat Cards ── */
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin-bottom: 2rem;
    }
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .stat-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        border-radius: 12px 12px 0 0;
    }
    .stat-card.blue::before   { background: #3B82F6; }
    .stat-card.violet::before { background: #8B5CF6; }
    .stat-card.teal::before   { background: #14B8A6; }
    .stat-card-icon {
        width: 32px;
        height: 32px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
    }
    .stat-card.blue   .stat-card-icon { background: #EFF6FF; }
    .stat-card.violet .stat-card-icon { background: #F5F3FF; }
    .stat-card.teal   .stat-card-icon { background: #F0FDFA; }
    .stat-card-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1C1C1E;
        line-height: 1;
        margin-bottom: 4px;
        letter-spacing: -0.03em;
    }
    .stat-card-label {
        font-size: 0.8rem;
        color: #8E8E93;
        font-weight: 400;
    }

    /* ── Feature Cards ── */
    .feature-card {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 10px;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        transition: border-color 0.15s;
    }
    .feature-card:hover {
        border-color: #C7C7CC;
    }
    .feature-icon-wrap {
        width: 36px;
        height: 36px;
        background: #F2F2F7;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-shrink: 0;
    }
    .feature-card-title {
        font-size: 0.88rem;
        font-weight: 600;
        color: #1C1C1E;
        margin: 0 0 3px;
    }
    .feature-card-desc {
        font-size: 0.82rem;
        color: #8E8E93;
        margin: 0;
        line-height: 1.45;
    }

    /* ── Section Headers ── */
    .section-header {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1C1C1E;
        margin: 1.5rem 0 1rem;
        letter-spacing: -0.01em;
    }

    /* ── Metric Row ── */
    .metric-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin: 1rem 0 1.5rem;
    }
    .metric-pill {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }
    .metric-pill-value {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1C1C1E;
        letter-spacing: -0.03em;
        line-height: 1;
        margin-bottom: 4px;
    }
    .metric-pill-label {
        font-size: 0.75rem;
        color: #8E8E93;
        font-weight: 400;
    }
    .metric-pill.easy   .metric-pill-value { color: #16A34A; }
    .metric-pill.medium .metric-pill-value { color: #CA8A04; }
    .metric-pill.hard   .metric-pill-value { color: #DC2626; }

    /* ── Tags / Badges ── */
    .badge {
        display: inline-block;
        padding: 2px 9px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-easy   { background: #DCFCE7; color: #166534; }
    .badge-medium { background: #FEF9C3; color: #854D0E; }
    .badge-hard   { background: #FEE2E2; color: #991B1B; }
    .badge-info   { background: #DBEAFE; color: #1E40AF; }
    .badge-gray   { background: #F2F2F7; color: #3C3C43; }

    /* ── Code / Arch block ── */
    .arch-block {
        background: #1C1C1E;
        border-radius: 12px;
        padding: 1.5rem 1.75rem;
        font-family: 'DM Mono', monospace;
        font-size: 0.78rem;
        color: #E5E5E7;
        line-height: 1.8;
        overflow-x: auto;
    }

    /* ── Status Badges (pipeline) ── */
    .pipeline-step {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 12px;
        font-size: 0.85rem;
        color: #1C1C1E;
    }
    .step-num {
        width: 24px;
        height: 24px;
        background: #1C1C1E;
        color: #FFFFFF;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: 600;
        flex-shrink: 0;
    }

    /* ── Sidebar brand ── */
    .sidebar-brand {
        padding: 0.5rem 0 1.25rem;
    }
    .sidebar-brand-name {
        font-size: 1.1rem;
        font-weight: 600;
        color: #FFFFFF !important;
        letter-spacing: -0.01em;
    }
    .sidebar-brand-sub {
        font-size: 0.72rem;
        color: #8E8E93 !important;
        margin-top: 2px;
    }

    /* ── Milestone badges ── */
    .milestone-badge {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #2C2C2E;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.78rem;
        color: #E5E5E7 !important;
    }
    .milestone-dot {
        width: 7px;
        height: 7px;
        background: #30D158;
        border-radius: 50%;
        flex-shrink: 0;
    }

    /* ── Streamlit overrides ── */
    .stButton > button {
        background: #1C1C1E !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.55rem 1.3rem !important;
        letter-spacing: 0.01em;
        transition: background 0.15s !important;
    }
    .stButton > button:hover {
        background: #3C3C3E !important;
    }
    [data-testid="stDownloadButton"] button {
        background: #FFFFFF !important;
        color: #1C1C1E !important;
        border: 1px solid #E5E5EA !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 8px !important;
        border: 1px solid #E5E5EA !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        background: #FFFFFF !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #3B82F6 !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 2px dashed #D1D1D6 !important;
        background: #FAFAFA !important;
    }
    .stSuccess {
        background: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        border-radius: 8px !important;
        color: #166534 !important;
    }
    .stWarning {
        background: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-radius: 8px !important;
        color: #000000 !important;
        font-weight: 500;
    }
    .stInfo {
        background: #EFF6FF !important;
        border: 1px solid #BFDBFE !important;
        border-radius: 8px !important;
        color: #000000 !important;
    }
    .stError {
        background: #FEF2F2 !important;
        border: 1px solid #FECACA !important;
        border-radius: 8px !important;
        color: #991B1B !important;
    }
    .stDataFrame {
        border-radius: 10px !important;
        overflow: hidden;
        border: 1px solid #E5E5EA !important;
    }
    .stMetric {
        background: #FFFFFF;
        border: 1px solid #E5E5EA;
        border-radius: 10px;
        padding: 0.75rem 1rem;
    }
    .stMetric label {
        font-size: 0.75rem !important;
        color: #8E8E93 !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 600 !important;
        color: #1C1C1E !important;
        letter-spacing: -0.03em;
    }
    .stExpander {
        border: 1px solid #E5E5EA !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
    }
    .stExpander summary {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #1C1C1E !important;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #1C1C1E !important;
        letter-spacing: -0.01em;
    }
    p, li, label, span {
        font-family: 'DM Sans', sans-serif;
    }
    .stSlider [data-testid="stSliderThumbValue"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
    }
    .stRadio [data-testid="stMarkdownContainer"] p {
        font-size: 0.85rem !important;
    }

    /* ── Matplotlib consistent style ── */
    .stPlotlyChart, .stPyplot { border-radius: 10px; overflow: hidden; }

    /* Fix ALL text elements inside expanders (report viewer) */
    .stExpander div[data-testid="stMarkdownContainer"] p,
    .stExpander div[data-testid="stMarkdownContainer"] li,
    .stExpander div[data-testid="stMarkdownContainer"] ul,
    .stExpander div[data-testid="stMarkdownContainer"] ol,
    .stExpander div[data-testid="stMarkdownContainer"] em,
    .stExpander div[data-testid="stMarkdownContainer"] strong,
    .stExpander div[data-testid="stMarkdownContainer"] span {
        color: #1C1C1E !important;
    }
    /* Fix li items globally in main content area too */
    div[data-testid="stMarkdownContainer"] li {
        color: #1C1C1E !important;
    }
    div[data-testid="stMarkdownContainer"] ul,
    div[data-testid="stMarkdownContainer"] ol {
        color: #1C1C1E !important;
    }

    /* Keep sidebar intact */
    section[data-testid="stSidebar"] * {
        color: #E5E5E7 !important;
    }
    div[data-testid="stAlert"] {
    color: black !important;
}

    /* Target inner text specifically */
    div[data-testid="stAlert"] * {
        color: black !important;
    }

    /* Extra safety (Streamlit uses this internally) */
    .stWarning div {
        color: black !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* 🔹 Slider labels (Easy Questions, Medium Questions, etc.) */
label, .stSlider label {
    color: #000000 !important;
}

/* 🔹 Slider numeric values (10, 20, 70) */
.stSlider [data-testid="stSliderThumbValue"] {
    color: #000000 !important;
    font-weight: 600;
}

/* 🔹 Slider min/max text if any */
.stSlider span {
    color: #000000 !important;
}

/* 🔹 Caption (Distribution — Easy: 10% ...) */
.stCaption, .stMarkdown p {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>

/* 🔹 Force ALL sidebar text to white */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* 🔹 Fix radio buttons text */
section[data-testid="stSidebar"] .stRadio label {
    color: #FFFFFF !important;
}

/* 🔹 Fix selected radio (active page) */
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #FFFFFF !important;
}

/* 🔹 Fix bottom footer text */
section[data-testid="stSidebar"] p {
    color: #FFFFFF !important;
}

/* 🔹 Fix small captions / spans */
section[data-testid="stSidebar"] span {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)

# ─── SVG Icon Library ──────────────────────────────────────────────────────
def icon(name, size=16, color="currentColor"):
    icons = {
        "upload": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>',
        "chart": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
        "users": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>',
        "trending": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
        "cpu": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
        "bot": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="16" x2="8" y2="16"/><line x1="16" y1="16" x2="16" y2="16"/></svg>',
        "home": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
        "layers": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
        "filter": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/></svg>',
        "zap": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
        "book": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
        "target": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
        "check": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
        "alert": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
        "download": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
        "search": f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>',
    }
    return icons.get(name, "")


def page_header(eyebrow, title, subtitle):
    st.markdown(f"""
    <div style="margin-bottom: 1.5rem;">
        <p class="page-eyebrow">{eyebrow}</p>
        <h1 class="page-title">{title}</h1>
        <p class="page-subtitle">{subtitle}</p>
    </div>
    <hr class="page-divider"/>
    """, unsafe_allow_html=True)


# ─── Matplotlib Theme ──────────────────────────────────────────────────────
def apply_plot_theme():
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.spines.left': True,
        'axes.spines.bottom': True,
        'axes.grid': True,
        'grid.color': '#F2F2F7',
        'grid.linewidth': 0.8,
        'axes.facecolor': '#FFFFFF',
        'figure.facecolor': '#FFFFFF',
        'axes.labelcolor': '#6B6B6B',
        'xtick.color': '#8E8E93',
        'ytick.color': '#8E8E93',
        'axes.titlecolor': '#1C1C1E',
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.spines.left': False,
    })
apply_plot_theme()

# ─── Sidebar ──────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <p class="sidebar-brand-name">ExamIQ</p>
    <p class="sidebar-brand-sub">Exam Analysis Platform</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Upload Data",
        "Difficulty Analysis",
        "Student Performance",
        "Visualizations",
        "Model Evaluation",
        "Assessment Assistant",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class="milestone-badge"><div class="milestone-dot"></div> Milestone 1 — ML Analytics</div>
<div class="milestone-badge"><div class="milestone-dot"></div> Milestone 2 — Agentic AI</div>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────
if "questions_df" not in st.session_state:
    st.session_state.questions_df = None
if "responses_df" not in st.session_state:
    st.session_state.responses_df = None


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
if page == "Home":
    page_header("Intelligent Analysis", "Exam Question Analysis System", "AI-driven platform to assess exam question quality, predict difficulty, and surface student performance patterns.")

    st.markdown(f"""
    <div class="stat-grid">
        <div class="stat-card blue">
            <div class="stat-card-icon">{icon("upload", 18, "#3B82F6")}</div>
            <div class="stat-card-value">CSV</div>
            <div class="stat-card-label">Upload exam questions and student responses</div>
        </div>
        <div class="stat-card violet">
            <div class="stat-card-icon">{icon("filter", 18, "#8B5CF6")}</div>
            <div class="stat-card-value">3-Class</div>
            <div class="stat-card-label">Easy · Medium · Hard difficulty prediction</div>
        </div>
        <div class="stat-card teal">
            <div class="stat-card-icon">{icon("trending", 18, "#14B8A6")}</div>
            <div class="stat-card-value">Agentic</div>
            <div class="stat-card-label">4-agent AI pipeline for assessment reports</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">Core Capabilities</p>', unsafe_allow_html=True)

    features = [
        ("layers",  "Difficulty Prediction",        "Classify questions as Easy, Medium, or Hard using Logistic Regression on TF-IDF features."),
        ("users",   "Student Performance Patterns",  "Analyze per-question response statistics and identify top and bottom performers."),
        ("search",  "Text Feature Extraction",        "TF-IDF based vectorization from raw question title and body text."),
        ("target",  "Model Evaluation",               "Accuracy, Precision, Recall, F1, and Confusion Matrix for the trained classifier."),
        ("chart",   "Visual Insights",                "Interactive charts for score distributions, time trends, and difficulty breakdowns."),
        ("bot",     "AI Assessment Assistant",        "4-agent pipeline: Analyzer → Retriever → Recommender → Reporter with PDF export."),
    ]

    for ic, title, desc in features:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon-wrap">{icon(ic, 17, "#3C3C43")}</div>
            <div>
                <p class="feature-card-title">{title}</p>
                <p class="feature-card-desc">{desc}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<p class="section-header">System Architecture</p>', unsafe_allow_html=True)
    st.markdown("""
    <div class="arch-block">Raw CSV Data
  └─ Text Cleaning & HTML Stripping
     └─ TF-IDF Feature Extraction
        └─ Logistic Regression Classifier
           ├─ Difficulty: Easy / Medium / Hard
           └─ Agent Pipeline
              ├─ Agent 1: Analyzer    — detect imbalance problems
              ├─ Agent 2: Retriever   — RAG pedagogical principles
              ├─ Agent 3: Recommender — LLM-generated suggestions
              └─ Agent 4: Reporter    — structured Markdown + PDF</div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE: Upload Data
# ══════════════════════════════════════════════
elif page == "Upload Data":
    page_header("Data Ingestion", "Upload Data", "Load your exam questions and student response files in CSV format.")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown(f"""
        <p class="section-header" style="display:flex;align-items:center;gap:8px;">
            {icon("book", 15, "#3C3C43")} Exam Questions
        </p>
        <p style="font-size:0.8rem;color:#8E8E93;margin-bottom:12px;">Expected columns: <code>Id</code>, <code>Title</code>, <code>Body</code>, <code>Score</code>, <code>Tags</code></p>
        """, unsafe_allow_html=True)
        questions_file = st.file_uploader("Upload Questions CSV", type=["csv"], key="q_upload", label_visibility="collapsed")
        if questions_file is not None:
            st.session_state.questions_df = pd.read_csv(questions_file, encoding="latin1", nrows=5000, on_bad_lines="skip")
            st.success(f"Loaded {len(st.session_state.questions_df):,} questions successfully")
            st.dataframe(st.session_state.questions_df.head(10), use_container_width=True)

    with col2:
        st.markdown(f"""
        <p class="section-header" style="display:flex;align-items:center;gap:8px;">
            {icon("users", 15, "#3C3C43")} Student Responses
        </p>
        <p style="font-size:0.8rem;color:#8E8E93;margin-bottom:12px;">Expected columns: <code>Id</code>, <code>ParentId</code>, <code>Body</code>, <code>Score</code></p>
        """, unsafe_allow_html=True)
        responses_file = st.file_uploader("Upload Responses CSV", type=["csv"], key="r_upload", label_visibility="collapsed")
        if responses_file is not None:
            st.session_state.responses_df = pd.read_csv(responses_file, encoding="latin1", nrows=5000, on_bad_lines="skip")
            st.success(f"Loaded {len(st.session_state.responses_df):,} responses successfully")
            st.dataframe(st.session_state.responses_df.head(10), use_container_width=True)

    st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
    if st.session_state.questions_df is not None:
        st.markdown('<p class="section-header">Questions — Summary Statistics</p>', unsafe_allow_html=True)
        st.dataframe(st.session_state.questions_df.describe(), use_container_width=True)
    if st.session_state.responses_df is not None:
        st.markdown('<p class="section-header">Responses — Summary Statistics</p>', unsafe_allow_html=True)
        st.dataframe(st.session_state.responses_df.describe(), use_container_width=True)


# ══════════════════════════════════════════════
# PAGE: Difficulty Analysis
# ══════════════════════════════════════════════
elif page == "Difficulty Analysis":
    page_header("ML Classification", "Difficulty Analysis", "Questions classified as Easy, Medium, or Hard based on score distribution percentiles.")

    if st.session_state.questions_df is not None:
        df = st.session_state.questions_df.copy()

        score_col = None
        for col in ["Score", "score", "OwnerUserId"]:
            if col in df.columns:
                score_col = col
                break

        if score_col and pd.api.types.is_numeric_dtype(df[score_col]):
            q_low  = df[score_col].quantile(0.33)
            q_high = df[score_col].quantile(0.66)

            df["Difficulty"] = pd.cut(
                df[score_col],
                bins=[-np.inf, q_low, q_high, np.inf],
                labels=["Hard", "Medium", "Easy"],
            )

            easy_count   = int((df["Difficulty"] == "Easy").sum())
            medium_count = int((df["Difficulty"] == "Medium").sum())
            hard_count   = int((df["Difficulty"] == "Hard").sum())

            difficulty_distribution = {
                "Easy": easy_count, "Medium": medium_count, "Hard": hard_count,
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

            # Metric row
            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="metric-pill-value">{len(df):,}</div>
                    <div class="metric-pill-label">Total Questions</div>
                </div>
                <div class="metric-pill easy">
                    <div class="metric-pill-value">{easy_count:,}</div>
                    <div class="metric-pill-label">Easy &nbsp;<span class="badge badge-easy">{difficulty_distribution['percentages']['Easy']}%</span></div>
                </div>
                <div class="metric-pill medium">
                    <div class="metric-pill-value">{medium_count:,}</div>
                    <div class="metric-pill-label">Medium &nbsp;<span class="badge badge-medium">{difficulty_distribution['percentages']['Medium']}%</span></div>
                </div>
                <div class="metric-pill hard">
                    <div class="metric-pill-value">{hard_count:,}</div>
                    <div class="metric-pill-label">Hard &nbsp;<span class="badge badge-hard">{difficulty_distribution['percentages']['Hard']}%</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if problems:
                st.markdown('<p class="section-header">Identified Issues</p>', unsafe_allow_html=True)
                for p in problems:
                    st.warning(p)
                st.info("Navigate to Assessment Assistant to run the full AI pipeline and generate improvement recommendations.")
            else:
                st.success("No issues detected — the exam appears well-balanced across difficulty levels.")

            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Difficulty Distribution</p>', unsafe_allow_html=True)

            diff_counts = df["Difficulty"].value_counts()
            fig, ax = plt.subplots(figsize=(7, 4))
            colors = ["#16A34A", "#CA8A04", "#DC2626"]
            vals   = diff_counts.reindex(["Easy", "Medium", "Hard"])
            bars   = ax.bar(vals.index, vals.values, color=colors, width=0.5, edgecolor="white", linewidth=1.5)
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(int(bar.get_height())), ha='center', va='bottom',
                        fontsize=10, color='#1C1C1E', fontweight='500')
            ax.set_ylabel("Questions", color="#6B6B6B")
            ax.set_title("Question Difficulty Distribution", color="#1C1C1E", fontsize=13, fontweight='600', pad=12)
            ax.set_ylim(0, vals.max() * 1.15)
            ax.spines['bottom'].set_color('#E5E5EA')
            ax.tick_params(axis='x', colors='#6B6B6B')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            col_exp, col_down = st.columns([3, 1])
            with col_exp:
                st.markdown('<p class="section-header">Classified Questions</p>', unsafe_allow_html=True)
                display_cols = [c for c in ["Id", "Title", "Score", score_col, "Difficulty"] if c in df.columns]
                display_cols = list(dict.fromkeys(display_cols))
                st.dataframe(df[display_cols].head(50), use_container_width=True)
            with col_down:
                st.markdown('<p class="section-header">Export</p>', unsafe_allow_html=True)
                st.download_button(
                    label=f"{icon('download', 14, '#1C1C1E')} Download JSON",
                    data=json.dumps(difficulty_distribution, indent=2),
                    file_name="difficulty_distribution.json",
                    mime="application/json",
                )
        else:
            st.warning("Could not find a numeric Score column. Please check your uploaded data.")
    else:
        st.info("Please upload questions data first from the Upload Data page.")


# ══════════════════════════════════════════════
# PAGE: Student Performance
# ══════════════════════════════════════════════
elif page == "Student Performance":
    page_header("Response Analytics", "Student Performance", "Analyze how students respond to exam questions and identify performance patterns.")

    if st.session_state.responses_df is not None:
        df = st.session_state.responses_df.copy()

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Responses", f"{len(df):,}")
        with c2:
            if "Score" in df.columns:
                df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
                st.metric("Average Score", f"{df['Score'].mean():.2f}")

        if "Score" in df.columns:
            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Response Score Distribution</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(df["Score"].clip(-10, 50), bins=30, color="#3B82F6", edgecolor="white", linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Response Scores", fontweight='600', pad=12)
            ax.spines['bottom'].set_color('#E5E5EA')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if "ParentId" in df.columns and "Score" in df.columns:
            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Per-Question Response Statistics</p>', unsafe_allow_html=True)
            per_q = df.groupby("ParentId")["Score"].agg(["mean", "count", "std"]).reset_index()
            per_q.columns = ["Question ID", "Avg Score", "Response Count", "Score Std Dev"]
            per_q = per_q.sort_values("Response Count", ascending=False)
            st.dataframe(per_q.head(50), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.markdown('<p class="section-header">Top Scoring Questions</p>', unsafe_allow_html=True)
                st.dataframe(
                    per_q.nlargest(10, "Avg Score")[["Question ID", "Avg Score", "Response Count"]],
                    use_container_width=True,
                )
            with col2:
                st.markdown('<p class="section-header">Lowest Scoring Questions</p>', unsafe_allow_html=True)
                st.dataframe(
                    per_q.nsmallest(10, "Avg Score")[["Question ID", "Avg Score", "Response Count"]],
                    use_container_width=True,
                )
    else:
        st.info("Please upload response data first from the Upload Data page.")


# ══════════════════════════════════════════════
# PAGE: Visualizations
# ══════════════════════════════════════════════
elif page == "Visualizations":
    page_header("Data Exploration", "Visualizations & Trends", "Interactive charts to explore question quality and performance over time.")

    if st.session_state.questions_df is not None:
        df = st.session_state.questions_df.copy()

        if "Score" in df.columns:
            st.markdown('<p class="section-header">Question Score Distribution</p>', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.hist(df["Score"].clip(-5, 100), bins=40, color="#8B5CF6", edgecolor="white", linewidth=0.8, alpha=0.85)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            ax.set_title("Distribution of Question Scores", fontweight='600', pad=12)
            ax.spines['bottom'].set_color('#E5E5EA')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        date_col = next((c for c in ["CreationDate", "creation_date"] if c in df.columns), None)
        if date_col:
            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Questions Over Time</p>', unsafe_allow_html=True)
            df["_date"]  = pd.to_datetime(df[date_col], errors="coerce")
            df["_month"] = df["_date"].dt.to_period("M")
            monthly = df.groupby("_month").size()
            fig, ax = plt.subplots(figsize=(10, 4))
            monthly.plot(kind="line", ax=ax, color="#F43F5E", linewidth=2.5)
            ax.fill_between(range(len(monthly)), monthly.values, alpha=0.08, color="#F43F5E")
            ax.set_xlabel("Month")
            ax.set_ylabel("Questions")
            ax.set_title("Questions Posted Over Time", fontweight='600', pad=12)
            ax.spines['bottom'].set_color('#E5E5EA')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        if "Score" in df.columns:
            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Score Distribution by Difficulty</p>', unsafe_allow_html=True)
            q_low  = df["Score"].quantile(0.33)
            q_high = df["Score"].quantile(0.66)
            df["Difficulty"] = pd.cut(
                df["Score"],
                bins=[-np.inf, q_low, q_high, np.inf],
                labels=["Hard", "Medium", "Easy"],
            )
            groups     = [df[df["Difficulty"] == d]["Score"].dropna() for d in ["Easy", "Medium", "Hard"]]
            box_colors = ["#16A34A", "#CA8A04", "#DC2626"]
            fig, ax = plt.subplots(figsize=(7, 4))
            bp = ax.boxplot(groups, patch_artist=True, notch=False,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(color="#C7C7CC"),
                            capprops=dict(color="#C7C7CC"),
                            flierprops=dict(marker='o', markersize=4, alpha=0.4))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.8)
            ax.set_xticklabels(["Easy", "Medium", "Hard"])
            ax.set_title("Score by Difficulty Category", fontweight='600', pad=12)
            ax.set_ylabel("Score")
            ax.spines['bottom'].set_color('#E5E5EA')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Please upload questions data first from the Upload Data page.")


# ══════════════════════════════════════════════
# PAGE: Model Evaluation
# ══════════════════════════════════════════════
elif page == "Model Evaluation":
    page_header("Machine Learning", "Model Evaluation", "Performance metrics for the Logistic Regression difficulty classifier.")

    MODEL_PATH      = "models/logistic_regression_model.pkl"
    VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model     = joblib.load(MODEL_PATH)
        tfidf_vec = joblib.load(VECTORIZER_PATH)

        st.success("Trained model loaded successfully")

        # Model info pills
        st.markdown(f"""
        <div class="metric-row" style="grid-template-columns: repeat(3, 1fr);">
            <div class="metric-pill">
                <div class="metric-pill-value" style="font-size:1rem;">Logistic Regression</div>
                <div class="metric-pill-label">Algorithm</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-value">{len(tfidf_vec.get_feature_names_out()):,}</div>
                <div class="metric-pill-label">TF-IDF Features</div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-value">{len(model.classes_)}</div>
                <div class="metric-pill-label">Classes ({", ".join(model.classes_)})</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Predict Question Difficulty</p>', unsafe_allow_html=True)
        user_question = st.text_area("Enter a question to predict its difficulty level:", placeholder="e.g. How do I reverse a linked list in Python?", height=110)

        if st.button("Run Prediction", type="primary"):
            if user_question.strip():
                cleaned      = clean_text_pipeline(user_question)
                q_tfidf      = tfidf_vec.transform([cleaned])
                prediction   = model.predict(q_tfidf)[0]
                probabilities = model.predict_proba(q_tfidf)[0]

                color_map = {"Easy": "#16A34A", "Medium": "#CA8A04", "Hard": "#DC2626"}
                badge_map = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}
                c = color_map.get(prediction, "#3B82F6")

                st.markdown(f"""
                <div style="background:#FFFFFF;border:1px solid #E5E5EA;border-left:4px solid {c};
                    border-radius:10px;padding:1.1rem 1.4rem;margin:1rem 0;display:flex;align-items:center;gap:12px;">
                    {icon("target", 20, c)}
                    <div>
                        <p style="margin:0;font-size:0.78rem;color:#8E8E93;font-weight:500;text-transform:uppercase;letter-spacing:0.05em;">Prediction</p>
                        <p style="margin:0;font-size:1.4rem;font-weight:600;color:{c};">{prediction}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                prob_df = pd.DataFrame({
                    "Difficulty": model.classes_,
                    "Confidence": [f"{p:.1%}" for p in probabilities],
                })
                st.dataframe(prob_df, use_container_width=False, hide_index=True)
            else:
                st.warning("Please enter a question first.")

        st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Performance Metrics</p>', unsafe_allow_html=True)

        CM_PATH   = "models/confusion_matrix.npy"
        metrics_data = None
        cm_matrix    = None
        cm_labels    = list(model.classes_)
        metrics_source = None

        # Always compute live on uploaded data (no hardcoded fallback)
        if st.session_state.questions_df is not None:
            df_eval = st.session_state.questions_df.copy()
            if "Score" in df_eval.columns and ("Title" in df_eval.columns or "Body" in df_eval.columns):
                with st.spinner("Computing metrics on uploaded data…"):
                    q_lo = df_eval["Score"].quantile(0.33)
                    q_hi = df_eval["Score"].quantile(0.66)
                    df_eval["_label"] = pd.cut(df_eval["Score"], bins=[-np.inf, q_lo, q_hi, np.inf], labels=["Hard", "Medium", "Easy"])
                    df_eval = df_eval.dropna(subset=["_label"])
                    title = df_eval["Title"].fillna("") if "Title" in df_eval.columns else pd.Series([""] * len(df_eval))
                    body  = df_eval["Body"].fillna("") if "Body" in df_eval.columns else pd.Series([""] * len(df_eval))
                    df_eval["_text"]    = title + " " + body
                    df_eval["_cleaned"] = df_eval["_text"].apply(clean_text_pipeline)
                    X_eval = tfidf_vec.transform(df_eval["_cleaned"])
                    y_true = df_eval["_label"].astype(str)
                    y_pred = model.predict(X_eval)
                    acc    = accuracy_score(y_true, y_pred)
                    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                    cm_matrix    = confusion_matrix(y_true, y_pred, labels=cm_labels)
                    metrics_data  = {"accuracy": acc, "report": report}
                    metrics_source = "Live evaluation on your uploaded questions data"

        if metrics_data is not None:
            st.caption(f"Source: {metrics_source}")
            acc    = metrics_data.get("accuracy", 0)
            report = metrics_data.get("report", {})
            macro  = report.get("macro avg", {})

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-pill">
                    <div class="metric-pill-value">{acc:.1%}</div>
                    <div class="metric-pill-label">Accuracy</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-value">{macro.get('precision', 0):.1%}</div>
                    <div class="metric-pill-label">Precision (macro)</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-value">{macro.get('recall', 0):.1%}</div>
                    <div class="metric-pill-label">Recall (macro)</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-value">{macro.get('f1-score', 0):.1%}</div>
                    <div class="metric-pill-label">F1 Score (macro)</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">Per-Class Breakdown</p>', unsafe_allow_html=True)
            per_class_rows = []
            for cls in ["Easy", "Medium", "Hard"]:
                if cls in report:
                    per_class_rows.append({
                        "Class": cls,
                        "Precision": f"{report[cls]['precision']:.3f}",
                        "Recall":    f"{report[cls]['recall']:.3f}",
                        "F1-Score":  f"{report[cls]['f1-score']:.3f}",
                        "Support":   int(report[cls]["support"]),
                    })
            if per_class_rows:
                st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

            if cm_matrix is not None:
                st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
                st.markdown('<p class="section-header">Confusion Matrix</p>', unsafe_allow_html=True)
                fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                im = ax_cm.imshow(cm_matrix, interpolation="nearest", cmap="Blues")
                plt.colorbar(im, ax=ax_cm, fraction=0.046)
                ax_cm.set_xticks(range(len(cm_labels)))
                ax_cm.set_yticks(range(len(cm_labels)))
                ax_cm.set_xticklabels(cm_labels, fontsize=10)
                ax_cm.set_yticklabels(cm_labels, fontsize=10)
                ax_cm.set_xlabel("Predicted", fontsize=11)
                ax_cm.set_ylabel("Actual",    fontsize=11)
                ax_cm.set_title("Confusion Matrix", fontsize=12, fontweight='600', pad=12)
                thresh = cm_matrix.max() / 2.0
                for i in range(cm_matrix.shape[0]):
                    for j in range(cm_matrix.shape[1]):
                        ax_cm.text(j, i, str(cm_matrix[i, j]),
                                   ha="center", va="center", fontsize=12, fontweight="600",
                                   color="white" if cm_matrix[i, j] > thresh else "#1C1C1E")
                plt.tight_layout()
                st.pyplot(fig_cm)
                plt.close()
        else:
            st.info("📂 Upload your questions CSV on the **Upload Data** page (needs a **Score** column + **Title** or **Body** column) to see live model performance metrics computed against your real data.")
    else:
        st.warning(f"No saved model found. Train and save the model from the notebook first.\nExpected paths: `{MODEL_PATH}` and `{VECTORIZER_PATH}`")


# ══════════════════════════════════════════════
# PAGE: Assessment Assistant
# ══════════════════════════════════════════════
elif page == "Assessment Assistant":
    page_header("Agentic AI Pipeline", "Assessment Assistant", "4-agent system: Analyzer — Retriever — Recommender — Reporter.")

    with st.expander("How the pipeline works", expanded=False):
        st.markdown("""
        **Agent 1 — Analyzer**: Detects difficulty imbalance problems from the distribution dict.

        **Agent 2 — Retriever**: RAG lookup to find relevant pedagogical principles for the identified issues.

        **Agent 3 — Recommender**: LLM (Gemini 2.0 Flash) generates 3 structured improvement recommendations. Falls back to rule-based output if API is unavailable.

        **Agent 4 — Reporter**: Formats a structured Markdown report suitable for PDF export.
        """)

    st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Step 1 — Difficulty Distribution Input</p>', unsafe_allow_html=True)

    has_m1_data = "difficulty_distribution" in st.session_state

    if has_m1_data:
        dist = st.session_state.difficulty_distribution
        st.success("Using difficulty data from Difficulty Analysis page")
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill easy">
                <div class="metric-pill-value">{dist['Easy']:,}</div>
                <div class="metric-pill-label">Easy &nbsp;<span class="badge badge-easy">{dist['percentages']['Easy']}%</span></div>
            </div>
            <div class="metric-pill medium">
                <div class="metric-pill-value">{dist['Medium']:,}</div>
                <div class="metric-pill-label">Medium &nbsp;<span class="badge badge-medium">{dist['percentages']['Medium']}%</span></div>
            </div>
            <div class="metric-pill hard">
                <div class="metric-pill-value">{dist['Hard']:,}</div>
                <div class="metric-pill-label">Hard &nbsp;<span class="badge badge-hard">{dist['percentages']['Hard']}%</span></div>
            </div>
            <div class="metric-pill">
                <div class="metric-pill-value">{dist['total']:,}</div>
                <div class="metric-pill-label">Total</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        difficulty_dict = {"Easy": dist["Easy"], "Medium": dist["Medium"], "Hard": dist["Hard"], "total": dist["total"]}
    else:
        st.info("No uploaded data detected. Use the sliders below or upload a CSV first.")
        col1, col2, col3 = st.columns(3)
        with col1:
            easy_count   = st.slider("Easy Questions",   0, 200, 10)
        with col2:
            medium_count = st.slider("Medium Questions", 0, 200, 20)
        with col3:
            hard_count   = st.slider("Hard Questions",   0, 200, 70)
        total_count = max(easy_count + medium_count + hard_count, 1)
        difficulty_dict = {"Easy": easy_count, "Medium": medium_count, "Hard": hard_count, "total": total_count}
        pct_easy = round(easy_count   / total_count * 100, 1)
        pct_med  = round(medium_count / total_count * 100, 1)
        pct_hard = round(hard_count   / total_count * 100, 1)
        st.caption(f"Distribution — Easy: {pct_easy}%   Medium: {pct_med}%   Hard: {pct_hard}%")

    st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
    st.markdown('<p class="section-header">Step 2 — Run Pipeline</p>', unsafe_allow_html=True)

    if st.button("Run AI Assessment Pipeline", type="primary"):
        try:
            from agents.analyzer  import run_analyzer_agent
            from agents.retriever import run_retriever_agent
            from agents.recommend import recommend_agent
            from agents.reporter  import generate_report

            topic_analysis = {}
            if "questions_df" in st.session_state and st.session_state.questions_df is not None:
                qdf = st.session_state.questions_df
                if "Title" in qdf.columns and "Score" in qdf.columns:
                    qdf_sorted = qdf.sort_values(by="Score")
                    extremes   = pd.concat([qdf_sorted.head(3), qdf_sorted.tail(2)]).drop_duplicates()
                    for _, row in extremes.iterrows():
                        topic_analysis[row["Title"]] = {
                            "score": float(row["Score"]) if pd.notnull(row["Score"]) else 0.0,
                            "difficulty": row.get("Difficulty", "Unknown")
                        }

            state = {"difficulty": difficulty_dict, "topic_analysis": topic_analysis}

            with st.status("Running 4-Agent Pipeline…", expanded=True) as pipeline_status:
                st.write("Agent 1 — Analyzer: Detecting difficulty problems…")
                state    = run_analyzer_agent(state)
                problems = state.get("problems", [])
                for p in problems:
                    st.write(f"  Issue: {p}")
                st.write(f"  {len(problems)} problem(s) identified")

                st.write("Agent 2 — Retriever: Fetching pedagogical principles via RAG…")
                state      = run_retriever_agent(state)
                principles = state.get("principles", [])
                for pr in principles:
                    st.write(f"  {pr[:90]}…")
                st.write(f"  {len(principles)} principle(s) retrieved")

                st.write("Agent 3 — Recommender: Generating recommendations via LLM…")
                state = recommend_agent(state)
                recs  = state.get("recommendations", [])
                st.write(f"  {len(recs)} recommendation(s) generated")

                st.write("Agent 4 — Reporter: Formatting structured report…")
                report_md = generate_report(state)
                st.write("  Report ready")

                pipeline_status.update(label="Pipeline Complete", state="complete", expanded=False)

            st.session_state.last_report       = report_md
            st.session_state.last_report_state = state

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)

    if "last_report" in st.session_state:
        st.markdown("<hr class='page-divider'/>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">Step 3 — Generated Report</p>', unsafe_allow_html=True)

        if "last_report_state" in st.session_state:
            s = st.session_state.last_report_state
            st.markdown(f"""
            <div class="metric-row" style="grid-template-columns: repeat(3, 1fr);">
                <div class="metric-pill">
                    <div class="metric-pill-value">{len(s.get('problems', []))}</div>
                    <div class="metric-pill-label">Problems Found</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-value">{len(s.get('principles', []))}</div>
                    <div class="metric-pill-label">Principles Retrieved</div>
                </div>
                <div class="metric-pill">
                    <div class="metric-pill-value">{len(s.get('recommendations', []))}</div>
                    <div class="metric-pill-label">Recommendations</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("View Full Report", expanded=True):
            st.markdown(st.session_state.last_report)

        try:
            from utils.pdf_export import create_pdf_report
            pdf_bytes = create_pdf_report(st.session_state.last_report)
            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name="Assessment_Report.pdf",
                mime="application/pdf",
            )
        except Exception as pdf_err:
            st.warning(f"PDF export unavailable: {pdf_err}")


st.sidebar.markdown("---")
st.sidebar.markdown('<p style="font-size:0.72rem;color:#636366;text-align:center;">ExamIQ — Intelligent Exam Analysis</p>', unsafe_allow_html=True)