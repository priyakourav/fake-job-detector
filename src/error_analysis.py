"""
Step 6: Error analysis on the LightGBM model.

Reads actual misclassified postings so you can see *what kind* of mistakes
the model makes — this is the difference between a tutorial clone and a
real portfolio project.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split

from features import get_structured_feature_matrix, get_text_series, get_label

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean.csv"
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "baseline_lightgbm.pkl"
THRESHOLD = 0.60  # chosen in evaluate.py as the best F1 threshold for LightGBM


def main():
    df = pd.read_csv(DATA_PATH)
    y = get_label(df)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

    bundle = joblib.load(MODEL_PATH)
    model, vectorizer, struct_cols = bundle["model"], bundle["vectorizer"], bundle["structured_cols"]

    X_text = vectorizer.transform(get_text_series(test_df))
    X_struct = csr_matrix(get_structured_feature_matrix(test_df)[struct_cols].values)
    X = hstack([X_text, X_struct]).tocsr()

    test_df = test_df.copy()
    test_df["pred_proba"] = model.predict_proba(X)[:, 1]
    test_df["pred"] = (test_df["pred_proba"] >= THRESHOLD).astype(int)

    fn = test_df[(test_df["fraudulent"] == 1) & (test_df["pred"] == 0)].sort_values("pred_proba")
    fp = test_df[(test_df["fraudulent"] == 0) & (test_df["pred"] == 1)].sort_values("pred_proba", ascending=False)

    print(f"\n{'='*60}")
    print(f"FALSE NEGATIVES (fraud postings the model MISSED): {len(fn)}")
    print(f"{'='*60}")
    for _, row in fn.head(5).iterrows():
        print(f"\n--- pred_proba={row['pred_proba']:.3f} ---")
        print(f"Title: {row['title']}")
        print(f"Description snippet: {row['description'][:300]}")

    print(f"\n{'='*60}")
    print(f"FALSE POSITIVES (genuine postings flagged as fraud): {len(fp)}")
    print(f"{'='*60}")
    for _, row in fp.head(5).iterrows():
        print(f"\n--- pred_proba={row['pred_proba']:.3f} ---")
        print(f"Title: {row['title']}")
        print(f"Description snippet: {row['description'][:300]}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total test set: {len(test_df)}")
    print(f"False negatives: {len(fn)} ({len(fn)/len(test_df):.1%} of test set)")
    print(f"False positives: {len(fp)} ({len(fp)/len(test_df):.1%} of test set)")


if __name__ == "__main__":
    main()