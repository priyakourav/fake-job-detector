"""
Step 5: Clean the raw Kaggle CSV and produce data/processed/clean.csv

Key decisions:
- We DON'T silently drop rows with missing text fields — missingness itself
  can be predictive (scam postings often skip company_profile, requirements).
  We add explicit `<field>_missing` flags instead.
- Text fields are concatenated into one `full_text` column for the baseline
  TF-IDF model in the next step.
- We keep the raw label column name `fraudulent` (0/1) untouched.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "fake_job_postings.csv"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "clean.csv"

TEXT_COLS = ["title", "company_profile", "description", "requirements", "benefits"]
CATEGORICAL_COLS = [
    "employment_type", "required_experience", "required_education",
    "industry", "function", "department",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Couldn't find {path}.\n"
            "Download 'fake_job_postings.csv' from the Kaggle dataset and place it there."
        )
    return pd.read_csv(path)


def add_missingness_flags(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS + CATEGORICAL_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def clean_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
            df[col] = df[col].str.replace(r"<[^>]+>", " ", regex=True)
            df[col] = df[col].str.replace(r"\s+", " ", regex=True).str.strip()
    return df


def build_full_text(df: pd.DataFrame) -> pd.DataFrame:
    df["full_text"] = (
        df["title"].fillna("") + " . "
        + df["company_profile"].fillna("") + " . "
        + df["description"].fillna("") + " . "
        + df["requirements"].fillna("") + " . "
        + df["benefits"].fillna("")
    ).str.strip()
    df["text_length"] = df["full_text"].str.len()
    df["word_count"] = df["full_text"].str.split().str.len()
    return df


def fill_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")
    return df


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw()
    print(f"Loaded {len(df):,} rows. Fraud rate: {df['fraudulent'].mean():.3%}")

    df = add_missingness_flags(df)
    df = clean_text_cols(df)
    df = build_full_text(df)
    df = fill_categoricals(df)

    before = len(df)
    df = df.drop_duplicates(subset=["full_text"])
    print(f"Dropped {before - len(df):,} duplicate postings.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote cleaned data to {OUT_PATH} ({len(df):,} rows).")


if __name__ == "__main__":
    main()