# Tensile Strength Prediction of Low-Alloy Steels

Predicting the tensile strength of low-alloy steels from chemical composition and processing parameters, using the MatNavi Low-Alloy Steels dataset — approached with both a machine learning and a metallurgical engineering lens.

🔗 **Live demo:** [propertypredictionofmaterial-qute6xffg7qpwwnyjbwdfe.streamlit.app](https://propertypredictionofmaterial-qute6xffg7qpwwnyjbwdfe.streamlit.app/)

---

## Motivation

Alloy design traditionally relies on empirical rules and expensive physical testing to relate composition and processing to mechanical properties. This project frames tensile strength prediction as a regression problem, using the interplay between chemistry (C, Mn, Si, Cr, Mo, etc.), processing (temperature, treatment), and microstructure-driven property outcomes — with model choices and feature behavior interpreted through a materials-science lens rather than treated as a black box.

## Dataset

- **Source:** [MatNavi Low-Alloy Steels dataset (Kaggle)](https://www.kaggle.com/)
- **Size:** 915 samples across 95 unique alloy compositions, each tested at multiple temperatures
- **Features:** Elemental composition (C, Si, Mn, P, S, Ni, Cr, Mo, Cu, V, etc.) and test temperature
- **Target:** Tensile strength

**Key data handling decision:** Since each alloy is tested repeatedly at different temperatures, a naive random train/test split leaks information (the same alloy composition can appear in both sets). This project uses **GroupShuffleSplit grouped by alloy code** so that no alloy composition appears in both training and test data — a more honest estimate of how the model generalizes to *unseen* alloys, not just unseen temperature readings.

## Exploratory Findings

- Individual elemental composition features showed weak linear correlation with tensile strength (~0.12), suggesting the composition–property relationship is non-additive and interaction-driven — consistent with how alloying elements act metallurgically (e.g., solid-solution strengthening, hardenability effects, carbide formation) rather than independently.
- **Temperature** was the strongest single linear predictor (r ≈ -0.33), reflecting the expected thermal softening behavior of steel.
- Significant multicollinearity was found among alloying elements (VIF: Mn ≈ 46.5, Si ≈ 21.2, Mo ≈ 20.1), which rules out plain linear regression as a reliable baseline and motivated regularization.

## Modeling & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Ridge Regression | 64.41 | 78.60 | 0.630 |
| Random Forest | 21.51 | 30.85 | 0.943 |
| **XGBoost** | **21.44** | **29.89** | **0.947** |

**Interpretation:** Ridge regression, even with regularization to handle multicollinearity, only explains ~63% of variance — confirming that tensile strength in this system is governed by non-linear interactions between composition and processing that a linear model structurally cannot capture. Tree-based ensembles close that gap almost entirely, with XGBoost as the best performer (R² = 0.947, MAE ≈ 21.4 MPa).

## Explainability (SHAP)

SHAP-based feature attribution is planned as the next step, to:
- Rank features by contribution to individual predictions
- Sanity-check model behavior against known metallurgical effects (e.g., Mo/Cr contribution to hardenability, Temperature's softening effect)
- Flag any learned relationships that *don't* align with metallurgical expectations, as a model-trust check

[SHAP Summary Plot](shap_summary.png)

## Deployment

An interactive **Streamlit app** (`app.py`) is deployed, allowing users to input alloy composition and processing parameters and get a live tensile strength prediction from the trained XGBoost model.

## Project Structure

```
├── app.py                          # Streamlit inference app
├── Property_prediction.ipynb       # EDA, feature engineering, modeling
├── random_forest_model.pkl         # Trained Random Forest model
├── xgb.pkl                         # Trained XGBoost model
├── requirements.txt                # Dependencies
└── *.csv                           # MatNavi raw dataset
```

## Tech Stack

Python · pandas · scikit-learn · XGBoost · SHAP (planned) · Streamlit

## Setup

```bash
git clone https://github.com/siddd776/Property_prediction_of_material.git
cd Property_prediction_of_material
pip install -r requirements.txt
streamlit run app.py
```

## Author

**Siddhant Agarwal** — Metallurgical & Materials Engineering, VNIT Nagpur

---

*Feedback and suggestions welcome — feel free to open an issue.*
