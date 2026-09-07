# Fake Job Posting Detector

Classifies job listings as real or fraudulent using NLP on posting text,
combined with structured metadata (has_logo, telecommuting, employment_type, etc).

Dataset: "Real or Fake Job Posting" (Kaggle) — ~18K postings, ~5% fraudulent.

---

## Project roadmap

- [x] **Step 0** — Environment setup
- [x] **Step 1** — Get the data from Kaggle
- [ ] **Step 2** — EDA
- [x] **Step 3** — Data prep + feature engineering
- [x] **Step 4** — Baseline model: TF-IDF + Logistic Regression / LightGBM
- [x] **Step 5** — Evaluate properly (precision/recall/F1/PR-AUC on fraud class)
- [ ] **Step 6** — Error analysis
- [ ] **Step 7** — (stretch) Fine-tune DistilBERT
- [ ] **Step 8** — Explainability (SHAP/LIME)
- [ ] **Step 9** — Demo app (Streamlit)
- [ ] **Step 10** — Write-up

---

## Setup

```bash
cd fake-job-detector
python -m venv venv
venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt
```

## Get the data

1. Download `fake_job_postings.csv` from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
2. Place it at `data/raw/fake_job_postings.csv`

## Run the pipeline

```bash
python src/data_prep.py

cd src
python train_baseline.py
python evaluate.py --model ../models/baseline_logreg.pkl
cd ..
```

---

## Results

| Model | Precision (fraud) | Recall (fraud) | F1 (fraud) | PR-AUC |
|---|---|---|---|---|
| TF-IDF + LogReg | 0.833 | 0.699 | 0.760 | *(fill from LightGBM run)* |
| TF-IDF + LightGBM | | | | |

## Notes on threshold choice

Chose threshold = 0.80 for the Logistic Regression baseline after sweeping
0.3–0.8. At this threshold: precision = 0.833, recall = 0.699, F1 = 0.760
(highest F1 among tested values). Optimized for a balanced trade-off since
both error types matter equally for this use case — flagging a genuine
posting as fraud (false positive, costs the user a real opportunity) is
just as bad as missing an actual scam (false negative, costs the user
safety).