# Micro-Lender Credit Scoring

## Problem
A micro-lending institution needs to assess borrower default risk using
applicant-level data, particularly for borrowers with limited formal credit
history. This project builds an ML model to estimate probability of default
and translate it into a risk band and lending recommendation.

## Dataset
Source: [Credit Risk Dataset (Kaggle)](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) — 
REAL public data, CC0 (Public Domain), ~32k individual loan records, included in this repo as `credit_risk_dataset.csv`.
Target: `loan_status` (0 = repaid, 1 = default).
Note: `loan_grade` and `loan_int_rate` were excluded as features — both are
assigned by the lender's own underwriting process and would leak information
about the outcome we're trying to predict.

## Approach
1. Data cleaning (invalid ages/employment length, missing value imputation)
2. EDA on default drivers
3. Feature engineering (income-to-employment ratio)
4. Models compared: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting
5. Final model: **Gradient Boosting** (ROC-AUC 0.879, PR-AUC 0.786)
6. Threshold tuned to 0.25 (business decision: prioritize catching defaulters)
7. Risk bands: Low / Medium / High / Very High, validated against actual default rates (6% → 19% → 34% → 92%)
8. SHAP explainability for transparency

## Results
See `model_comparison.csv` and `threshold_analysis.csv`.

## Nigerian Context Adaptation

This project uses a public US-centric credit dataset (Kaggle, CC0) as the
closest available proxy for individual credit-risk modeling, since no
open, purpose-built micro-lending dataset for the Nigerian market currently
exists. To adapt this approach for a real Nigerian micro-lender, the
following changes would be needed:

- **Credit history proxy**: `cb_person_cred_hist_length` (years of formal
  credit history) has limited relevance for many Nigerian borrowers who
  are financially included but credit-invisible. A real deployment would
  substitute mobile money transaction history (e.g. Opay, PalmPay, Moniepoint),
  BVN-linked repayment records, or cooperative/thrift (ajo/esusu) contribution
  history as alternative signals of repayment reliability.
- **Income representation**: Income here is a raw USD figure; a Nigerian
  version would use NGN-denominated income bands and account for informal,
  irregular, or seasonal income common among small traders and artisans,
  rather than assuming steady formal-sector pay.
- **Employment/housing categories**: `person_home_ownership` (RENT/OWN/
  MORTGAGE) reflects US housing finance norms. A Nigerian equivalent would
  need categories reflecting informal employment (trading, artisanal work,
  gig income) and housing arrangements (family house, informal rental)
  more common locally.
- **Loan sizing**: Loan amounts in this dataset (thousands of USD) are
  far larger than typical Nigerian micro-loans (often ₦20,000–₦500,000).
  A production model would need to be retrained on locally scaled amounts.

The modeling approach — leakage-aware feature selection, threshold-based
risk banding, and SHAP-based explainability — is directly transferable;
only the underlying data and feature definitions would need to be
Nigeria-specific.

## Important Disclaimer
This model outputs a **risk estimate**, not a lending decision. Real-world
deployment requires human review, regulatory compliance, affordability
checks, and fraud screening alongside model output.

## Files
- `credit_scoring_capstone.ipynb` — full notebook
- `credit_risk_model.pkl` — trained model
- `scored_applicants_sample.csv` — example scored applicants
- `README.md` — this file