"""
Step 6a: Feature engineering.

Turns the cleaned dataframe into an (X_text, X_structured) pair the baseline
model can use. Kept separate from data_prep.py so you can iterate on features
without re-running the (slower) text cleaning step.
"""
import pandas as pd
import numpy as np

STRUCTURED_NUMERIC = [
    "telecommuting", "has_company_logo", "has_questions",
    "text_length", "word_count",
]

MISSING_FLAG_COLS = [
    "title_missing", "company_profile_missing", "description_missing",
    "requirements_missing", "benefits_missing",
    "employment_type_missing", "required_experience_missing",
    "required_education_missing", "industry_missing", "function_missing",
    "department_missing",
]

# hand-picked "scam smell" signals — cheap to compute, often show up as
# important features, and are easy to explain to a non-technical reader
SUSPICIOUS_KEYWORDS = [
    "no experience", "work from home", "quick money", "wire transfer",
    "processing fee", "click here", "urgent", "immediate start",
    "earn $", "guaranteed",
]


def add_keyword_features(df: pd.DataFrame, text_col: str = "full_text") -> pd.DataFrame:
    lowered = df[text_col].str.lower()
    for kw in SUSPICIOUS_KEYWORDS:
        col_name = "kw_" + kw.replace(" ", "_").replace("$", "dollar")
        df[col_name] = lowered.str.contains(kw, regex=False).astype(int)
    return df


def get_structured_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    df = add_keyword_features(df)
    kw_cols = [c for c in df.columns if c.startswith("kw_")]
    cols = STRUCTURED_NUMERIC + MISSING_FLAG_COLS + kw_cols
    cols = [c for c in cols if c in df.columns]
    return df[cols].fillna(0)


def get_text_series(df: pd.DataFrame) -> pd.Series:
    return df["full_text"]


def get_label(df: pd.DataFrame) -> pd.Series:
    return df["fraudulent"]