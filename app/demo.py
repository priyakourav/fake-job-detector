"""
Streamlit demo app — dark dashboard UI (emerald green accent theme).

Run with: streamlit run demo.py  (from inside the app/ folder)
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import hstack, csr_matrix

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline_logreg.pkl"

st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🛡️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------- Dark dashboard theme ----------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 20% 0%, #16221F 0%, #0B0F0E 45%, #060807 100%) !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 780px;
    }

    .app-header {
        text-align: center;
        margin-bottom: 0.25rem;
    }
    .app-header h1 {
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FFFFFF 0%, #A7F3D0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        text-align: center;
        color: #8A9A95;
        font-size: 0.95rem;
        margin-bottom: 2rem;
    }

    .glass-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(52,211,153,0.15);
        border-radius: 18px;
        padding: 1.6rem 1.6rem 1.2rem 1.6rem;
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }

    label, .stTextInput label, .stTextArea label,
    [data-testid="stWidgetLabel"] p {
        color: #C9D6D1 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
    }

    /* Force-override BaseWeb's default white input backgrounds */
    .stTextArea textarea,
    .stTextInput input,
    div[data-baseweb="textarea"],
    div[data-baseweb="textarea"] textarea,
    div[data-baseweb="input"],
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] input {
        background-color: #131C1A !important;
        border-radius: 12px !important;
        border: 1.5px solid rgba(255,255,255,0.12) !important;
        color: #E5F5EF !important;
        font-size: 0.95rem !important;
        -webkit-text-fill-color: #E5F5EF !important;
        caret-color: #34D399 !important;
    }
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #6B7C77 !important;
        -webkit-text-fill-color: #6B7C77 !important;
        opacity: 1 !important;
    }
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #34D399 !important;
        box-shadow: 0 0 0 3px rgba(52,211,153,0.15) !important;
    }
    .stTextArea > div, .stTextInput > div {
        background-color: transparent !important;
        border: none !important;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #34D399 0%, #059669 100%);
        color: #05221A;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 1.4rem;
        font-weight: 700;
        font-size: 0.95rem;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 20px rgba(52,211,153,0.3);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(52,211,153,0.45);
    }
    div.stButton > button p {
        color: #05221A !important;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.6rem 1.8rem;
        margin-top: 1.5rem;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        backdrop-filter: blur(10px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
    }
    .score-label {
        font-size: 0.8rem;
        color: #8A9A95;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .score-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin-top: 0.3rem;
    }
    .verdict-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-weight: 700;
        font-size: 0.85rem;
        margin-top: 0.7rem;
        border: 1px solid transparent;
    }
    .badge-high { background: rgba(248,113,113,0.12); color: #F87171; border-color: rgba(248,113,113,0.3); }
    .badge-mod  { background: rgba(251,191,36,0.12); color: #FBBF24; border-color: rgba(251,191,36,0.3); }
    .badge-low  { background: rgba(52,211,153,0.12); color: #34D399; border-color: rgba(52,211,153,0.3); }

    .result-message {
        color: #A9B8B2;
        margin-top: 0.9rem;
        margin-bottom: 0;
        font-size: 0.92rem;
    }

    .section-title {
        font-weight: 700;
        font-size: 1.02rem;
        color: #E5F5EF;
        margin-top: 1.8rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stAlert"] {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        color: #C9D6D1;
    }

    .stCaption, [data-testid="stCaptionContainer"] {
        color: #5C6E68 !important;
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="app-header">
    <h1>🛡️ Fake Job Posting Detector</h1>
</div>
<div class="app-subtitle">
    Paste a job description to check it against patterns learned from 15,000+ real and fraudulent listings.
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


bundle = load_model()

if bundle is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Run these first:\n"
        "1. `python src/data_prep.py`\n"
        "2. `python src/train_baseline.py` (from inside src/)"
    )
    st.stop()

model, vectorizer = bundle["model"], bundle["vectorizer"]

title = st.text_input("Job title (optional)", placeholder="e.g. Remote Data Entry Specialist")
description = st.text_area(
    "Job description",
    height=220,
    placeholder="Paste the full job posting text here...",
)
analyze_clicked = st.button("🔍  Analyze Posting", type="primary")
if analyze_clicked and description.strip():
    full_text = f"{title} . {description}"

    n_struct = len(bundle["structured_cols"])
    X_text = vectorizer.transform([full_text])
    X_struct = csr_matrix(np.zeros((1, n_struct)))
    X = hstack([X_text, X_struct]).tocsr()

    proba = model.predict_proba(X)[0, 1]

    if proba >= 0.7:
        badge_class, badge_text, score_color = "badge-high", "⚠️  High Risk", "#F87171"
        message = "This posting shares strong patterns with known fraudulent listings."
    elif proba >= 0.4:
        badge_class, badge_text, score_color = "badge-mod", "🟡  Moderate Risk", "#FBBF24"
        message = "Some suspicious signals present. Verify the company independently before proceeding."
    else:
        badge_class, badge_text, score_color = "badge-low", "✅  Low Risk", "#34D399"
        message = "Looks consistent with legitimate postings in the training data."

    st.markdown(f"""
    <div class="result-card">
        <div class="score-label">Fraud Probability</div>
        <div class="score-value" style="color:{score_color};">{proba:.1%}</div>
        <div class="verdict-badge {badge_class}">{badge_text}</div>
        <p class="result-message">{message}</p>
    </div>
    """, unsafe_allow_html=True)

    feature_names = vectorizer.get_feature_names_out()
    n_text_features = len(feature_names)
    coefs = model.coef_[0][:n_text_features]

    text_vec = X_text.toarray().ravel()
    present_idx = np.nonzero(text_vec)[0]
    contributions = text_vec[present_idx] * coefs[present_idx]

    top_n = min(10, len(present_idx))
    order = np.argsort(contributions)[::-1][:top_n]

    contrib_df = pd.DataFrame({
        "phrase": feature_names[present_idx[order]],
        "contribution": contributions[order],
    })
    contrib_df = contrib_df[contrib_df["contribution"] > 0]

    st.markdown('<div class="section-title">What drove this score</div>', unsafe_allow_html=True)
    if len(contrib_df):
        st.bar_chart(contrib_df.set_index("phrase"), color="#34D399")
    else:
        st.info("No strong fraud-associated phrases detected in this text.")

    st.caption(
        "Model: TF-IDF + Logistic Regression trained on the Kaggle "
        "'Real or Fake Job Posting' dataset · Educational/portfolio use only — "
        "always verify a company independently before applying or sharing personal info."
    )
elif analyze_clicked:
    st.warning("" \
    "Please paste a job description first.")