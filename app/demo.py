"""
Streamlit demo app.

Run with: streamlit run demo.py  (from inside the app/ folder)

Paste a job posting, get a fraud probability plus the top words/phrases
that pushed the score up. Uses the Logistic Regression model because it's
directly interpretable (raw coefficients) -- that's what powers the
"what drove this score" explanation below without needing SHAP.
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

st.set_page_config(page_title="Fake Job Posting Detector", page_icon="🕵️", layout="centered")
st.title("🕵️ Fake Job Posting Detector")
st.caption("Paste a job description below. This is a portfolio demo, not a substitute for your own judgment.")


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

title = st.text_input("Job title (optional)", "")
description = st.text_area("Job description / full posting text", height=250,
                            placeholder="Paste the job posting text here...")

if st.button("Analyze", type="primary") and description.strip():
    full_text = f"{title} . {description}"

    n_struct = len(bundle["structured_cols"])
    X_text = vectorizer.transform([full_text])
    X_struct = csr_matrix(np.zeros((1, n_struct)))
    X = hstack([X_text, X_struct]).tocsr()

    proba = model.predict_proba(X)[0, 1]

    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Fraud probability", f"{proba:.1%}")
    with col2:
        if proba >= 0.7:
            st.error("⚠️ High risk — this posting shares strong patterns with known fraudulent listings.")
        elif proba >= 0.4:
            st.warning("🟡 Moderate risk — some suspicious signals present. Verify the company independently.")
        else:
            st.success("✅ Low risk — looks consistent with legitimate postings in the training data.")

    st.subheader("What drove this score")
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

    if len(contrib_df):
        st.bar_chart(contrib_df.set_index("phrase"))
    else:
        st.write("No strong fraud-associated phrases detected in this text.")

    st.caption(
        "Model: TF-IDF + Logistic Regression trained on the Kaggle "
        "'Real or Fake Job Posting' dataset. Educational/portfolio use only — "
        "always verify a company independently before applying or sharing personal info."
    )