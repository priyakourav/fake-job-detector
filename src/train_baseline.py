"""
Step 6b: Train baseline models.

Two models, both using TF-IDF text features + the structured/keyword
features from features.py:
  1. Logistic Regression (interpretable, fast, good baseline)
  2. LightGBM (usually a bit stronger, handles the structured features natively)

class_weight="balanced" / scale_pos_weight are used instead of naive
oversampling — with ~5% positive rate, SMOTE-on-TF-IDF tends to create
nonsense synthetic text vectors. Balanced class weighting is the more
defensible choice here and is easy to justify in an interview.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, average_precision_score, f1_score
)

import lightgbm as lgb

from features import get_structured_feature_matrix, get_text_series, get_label

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Run `python src/data_prep.py` first."
        )
    return pd.read_csv(DATA_PATH)


def build_splits(df, test_size=0.2, random_state=42):
    y = get_label(df)
    idx_train, idx_test = train_test_split(
        df.index, test_size=test_size, random_state=random_state, stratify=y
    )
    return df.loc[idx_train], df.loc[idx_test]


def vectorize_text(train_text, test_text, max_features=15000):
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=3,
        stop_words="english",
        sublinear_tf=True,
    )
    X_train = vec.fit_transform(train_text)
    X_test = vec.transform(test_text)
    return X_train, X_test, vec


def train_logreg(X_train, y_train):
    clf = LogisticRegression(
        max_iter=2000, class_weight="balanced", C=1.0, solver="liblinear"
    )
    clf.fit(X_train, y_train)
    return clf


def train_lightgbm(X_train, y_train):
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    clf = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate(name, clf, X_test, y_test):
    proba = clf.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    print(f"\n=== {name} (threshold=0.5) ===")
    print(classification_report(y_test, preds, target_names=["real", "fraudulent"]))
    pr_auc = average_precision_score(y_test, proba)
    f1_fraud = f1_score(y_test, preds, pos_label=1)
    print(f"PR-AUC: {pr_auc:.4f} | F1 (fraud class): {f1_fraud:.4f}")
    return {"pr_auc": pr_auc, "f1_fraud": f1_fraud}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_features", type=int, default=15000)
    args = parser.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)
    df = load_data()
    train_df, test_df = build_splits(df)

    print(f"Train: {len(train_df):,} rows | Test: {len(test_df):,} rows")
    print(f"Train fraud rate: {get_label(train_df).mean():.3%} | "
          f"Test fraud rate: {get_label(test_df).mean():.3%}")

    Xtr_text, Xte_text, vectorizer = vectorize_text(
        get_text_series(train_df), get_text_series(test_df), args.max_features
    )

    Xtr_struct = csr_matrix(get_structured_feature_matrix(train_df).values)
    Xte_struct = csr_matrix(get_structured_feature_matrix(test_df).values)

    X_train = hstack([Xtr_text, Xtr_struct]).tocsr()
    X_test = hstack([Xte_text, Xte_struct]).tocsr()
    y_train, y_test = get_label(train_df), get_label(test_df)

    results = {}

    logreg = train_logreg(X_train, y_train)
    results["logreg"] = evaluate("Logistic Regression", logreg, X_test, y_test)
    joblib.dump(
        {"model": logreg, "vectorizer": vectorizer,
         "structured_cols": get_structured_feature_matrix(train_df).columns.tolist()},
        MODELS_DIR / "baseline_logreg.pkl",
    )

    lgbm = train_lightgbm(X_train, y_train)
    results["lightgbm"] = evaluate("LightGBM", lgbm, X_test, y_test)
    joblib.dump(
        {"model": lgbm, "vectorizer": vectorizer,
         "structured_cols": get_structured_feature_matrix(train_df).columns.tolist()},
        MODELS_DIR / "baseline_lightgbm.pkl",
    )

    print("\n=== Summary ===")
    for name, r in results.items():
        print(f"{name:12s} PR-AUC={r['pr_auc']:.4f}  F1(fraud)={r['f1_fraud']:.4f}")

    print(f"\nModels saved to {MODELS_DIR}/")


if __name__ == "__main__":
    main()