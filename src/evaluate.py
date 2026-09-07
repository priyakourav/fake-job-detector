"""
Step 7: Standalone evaluation + figures for a saved model.

Usage:
    python evaluate.py --model ../models/baseline_logreg.pkl

Produces:
    reports/figures/confusion_matrix.png
    reports/figures/pr_curve.png
And prints a precision/recall table across a few candidate thresholds so
you can make (and justify) a deliberate threshold choice instead of
defaulting to 0.5.
"""
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, PrecisionRecallDisplay,
    precision_score, recall_score, f1_score,
)

from features import get_structured_feature_matrix, get_text_series, get_label

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean.csv"
FIG_DIR = Path(__file__).resolve().parents[1] / "reports" / "figures"


def load_test_split(random_state=42, test_size=0.2):
    df = pd.read_csv(DATA_PATH)
    y = get_label(df)
    _, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=y)
    return test_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a saved .pkl from train_baseline.py")
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.model)
    model, vectorizer, struct_cols = bundle["model"], bundle["vectorizer"], bundle["structured_cols"]

    test_df = load_test_split()
    X_text = vectorizer.transform(get_text_series(test_df))
    X_struct = csr_matrix(get_structured_feature_matrix(test_df)[struct_cols].values)
    X = hstack([X_text, X_struct]).tocsr()
    y_true = get_label(test_df)

    proba = model.predict_proba(X)[:, 1]

    print("\nThreshold sweep (fraud class):")
    print(f"{'thresh':>8} {'precision':>10} {'recall':>8} {'f1':>8}")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (proba >= t).astype(int)
        p = precision_score(y_true, preds, pos_label=1, zero_division=0)
        r = recall_score(y_true, preds, pos_label=1, zero_division=0)
        f1 = f1_score(y_true, preds, pos_label=1, zero_division=0)
        print(f"{t:8.2f} {p:10.3f} {r:8.3f} {f1:8.3f}")

    preds_05 = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_05)
    disp = ConfusionMatrixDisplay(cm, display_labels=["real", "fraudulent"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (threshold=0.5)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_true, proba)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("Precision-Recall Curve (fraud class)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "pr_curve.png", dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()