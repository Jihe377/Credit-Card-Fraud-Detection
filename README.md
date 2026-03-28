# Credit Card Fraud Detection
### A Risk Control Product Perspective

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [The Decision Architecture](#the-decision-architecture)
- [Dataset & Exploratory Findings](#dataset--exploratory-findings)
- [Methodology](#methodology)
- [Results & Risk Signal Analysis](#results--risk-signal-analysis)
- [From Model Output to Operational Decision](#from-model-output-to-operational-decision)
- [Limitations & Open Questions](#limitations--open-questions)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [References](#references)

---

## Problem Statement
 
Every credit card transaction triggers a decision that must be made in under 300 milliseconds: **pass, flag for review, or block**. The system making this decision operates under three structural constraints that make it fundamentally different from most classification problems:
 
**1. Extreme class imbalance.** Fraud accounts for roughly 0.1–0.3% of transactions in real-world payment networks. On this dataset, the ratio is 577:1. A classifier that predicts every transaction as legitimate achieves 99.83% accuracy while catching zero fraud — accuracy is therefore not a meaningful metric.
 
**2. Asymmetric error costs.** The two types of errors carry very different consequences:
 
| Error | What happened | Consequence |
|---|---|---|
| False Negative | Fraudulent transaction passed | Direct financial loss + downstream dispute cost + regulatory exposure |
| False Positive | Legitimate transaction blocked | User friction + potential churn + support cost |
 
In most payment contexts, a missed fraud costs 10–50× more than a false positive. This asymmetry means the decision threshold between "flag" and "pass" is not a technical parameter — it is a business decision that encodes the organization's risk appetite.
 
**3. Adversarial dynamics.** Fraud patterns evolve continuously as attackers probe for weaknesses in detection systems. A model that achieves strong performance today degrades as behavioral patterns shift. This makes monitoring and retraining part of the system design, not an afterthought.
 
The goal of this project is to build and evaluate a fraud detection model while explicitly analyzing the relationship between model outputs and operational risk decisions.
 
---

## The Decision Architecture
 
No production fraud system is a single model. In practice, transaction decisioning is layered:

```
Transaction arrives
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 1: Rules Engine                  Latency: <10ms │
│  Hard blocklist + velocity checks                     │
│  → High-confidence known fraud patterns               │
│    Output: BLOCK (immediate)                          │
└──────────────────────────────────────────────────────┘
        │ (passes through)
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 2: ML Scoring Model           Latency: 50–200ms │
│  XGBoost risk score + dynamic threshold               │
│  → Ambiguous transactions needing probabilistic eval  │
│    Output: PASS / BLOCK / REVIEW_QUEUE                │
└──────────────────────────────────────────────────────┘
        │ (uncertain or high-value cases)
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 3: Human Review Queue         Latency: min–hrs  │
│  Analyst-reviewed, prioritized by risk score          │
│  → Low-confidence, high-value, or novel pattern cases │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Feedback Loop                                        │
│  Dispute outcomes → labeled ground truth              │
│  → Retraining data for model improvement              │
└──────────────────────────────────────────────────────┘
```
 
This project focuses on **Tier 2**: the ML scoring model that handles the ambiguous middle ground between obvious fraud and obviously legitimate transactions. The model output is a risk score, not a binary decision — the threshold that converts that score into an action is set externally based on business context.
 
---

## Dataset & Exploratory Findings
 
| Property | Value |
|---|---|
| Source | [Kaggle — ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Time window | Two days of European cardholder transactions (Sep 2013) |
| Features | V1–V28 (PCA-anonymized) + `Amount` + `Time` |
| Class ratio | ~577:1 (legitimate : fraud) |

### Exploratory Findings
 
**Transaction amount.** Fraudulent transactions skew toward lower amounts (median ~$22) compared to legitimate transactions (median ~$62). This pattern is consistent with card testing behavior — attackers often initiate small-value transactions to verify that a stolen card is active before attempting higher-value fraud.
 
**Temporal distribution.** Fraud events show mild clustering in the early morning hours (roughly 0:00–4:00), when automated fraud detection and customer self-monitoring are both at their lowest. This temporal signal is weak individually but contributes to multi-feature fraud scoring.
 
**Feature separability.** Despite PCA anonymization, several components — most notably V4, V11, V14, and V17 — show substantially different distributions between fraud and legitimate classes. V14 in particular exhibits the highest individual feature importance in the trained XGBoost model (see [Results](#-results--risk-signal-analysis)).
 
**What PCA anonymization means for analysis.** The V1–V28 features are principal components of the original transaction features, which the dataset publisher cannot disclose for privacy reasons. This means individual features cannot be interpreted in isolation — we can observe that V14 is a strong fraud predictor, but cannot directly infer which original behavioral attributes it captures. This is a limitation that affects model explainability in deployment (see [Limitations](#-limitations--open-questions)).
 
---

## Methodology
 
### Evaluation metric selection
 
Given the class imbalance and asymmetric cost structure, two metrics govern model evaluation:
 
**Recall on the fraud class** is the primary metric. It measures what fraction of actual fraud events the model correctly identifies. A missed fraud translates directly to financial loss.
 
**AUC-PR (Area Under the Precision-Recall Curve)** is used over AUC-ROC as the secondary metric. AUC-ROC averages performance across all classification thresholds, including regions where the base rate is so low that the metric is dominated by true negative performance. AUC-PR focuses specifically on the minority class, providing a more honest picture of model utility in imbalanced settings.
 
Accuracy is reported but not used for model selection.
### Handling class imbalance: SMOTE
 
Two approaches address the 577:1 imbalance:
 
- **Threshold adjustment**: shifting the decision boundary at inference time without changing the model. Fast but does not improve the model's learned representation of fraud patterns.
- **SMOTE (Synthetic Minority Oversampling Technique)**: generating synthetic fraud samples during training by interpolating between existing fraud instances in feature space. Higher training cost, but produces better generalization to novel fraud patterns.
 
This project applies SMOTE during training, with threshold tuning available as a separate inference-time control. Critically, **SMOTE is applied only to the training split** — applying it before the train/test split would allow synthetic samples derived from test-set points to contaminate training, producing inflated metrics that overestimate real-world performance.
 
### Model selection rationale
 
Four model families are evaluated. The selection criterion is not raw benchmark performance but fitness for the operational constraints of fraud detection:
 
**Logistic Regression** serves as the baseline. It is fast, interpretable, and well-understood. Its limitations in handling nonlinear feature interactions make it a ceiling-setter for simple approaches rather than a deployment candidate.
 
**Random Forest** handles nonlinear interactions and is naturally robust to outliers. Its ensemble structure provides some protection against overfitting on the highly imbalanced training data.
 
**XGBoost** is the primary candidate for two reasons beyond performance. First, in financial risk decisions, a declined transaction typically requires a documented reason — for the user, for fraud analysts, and in many markets, for regulators. XGBoost's feature importance scores provide this audit trail; neural networks do not. Second, XGBoost is natively robust to missing values and more resistant to feature distribution drift, which matters in production where fraud patterns shift continuously.
 
**SVM** is included for comparison on a problem where the decision boundary between fraud and legitimate transactions may not be linearly separable in the original feature space.
 
### Pipeline
 
```
Raw Data (284,807 transactions)
        │
        ▼
  Exploratory Analysis
  → Class distribution, Amount/Time distributions,
    feature correlation and separability by class
        │
        ▼
  Preprocessing
  → StandardScaler on Amount and Time
    Stratified train/test split (80/20)
        │
        ▼
  Imbalance Handling (training set only)
  → SMOTE applied after split
        │
        ▼
  Model Training & Comparison
  → XGBoost, Random Forest, Logistic Regression, SVM
    5-fold stratified cross-validation
        │
        ▼
  Evaluation
  → Recall, Precision, F1, AUC-ROC, AUC-PR
    Precision-Recall curve, Confusion matrix
    Feature importance (XGBoost)
```
---
 
## Results & Risk Signal Analysis
 
### Model comparison
 
| Model | Recall ↑ | Precision | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| **XGBoost** | **91.5%** | 88.2% | 89.8% | 0.974 | 0.891 |
| Random Forest | 89.1% | 90.4% | 89.7% | 0.968 | 0.874 |
| Logistic Regression | 82.3% | 85.7% | 84.0% | 0.943 | 0.832 |
| SVM | 84.6% | 83.1% | 83.8% | 0.951 | 0.848 |
 
XGBoost achieves the highest Recall (91.5%) and AUC-PR (0.891). Random Forest is competitive on Precision and F1, and may be preferable in scenarios where false positive rate reduction is the primary business concern.
 
### What 91.5% Recall means operationally
 
Out of 100 fraudulent transactions processed by the model:
- **~92 are correctly scored above threshold** and routed to block or human review
- **~8 are missed** and will result in direct financial loss
 
Whether this is operationally sufficient depends on the transaction value distribution:
 
| Transaction context | Assessment |
|---|---|
| Low-value transactions (< $50) | Generally sufficient — bounded loss per miss, false positives hurt conversion more |
| Mid-value transactions ($50–$500) | Marginal — consider tiered review thresholds for uncertain cases |
| High-value / wire transfers (> $500) | Insufficient as standalone — requires additional review layer for flagged cases |
 
This illustrates a core principle of fraud risk management: **a single model threshold is never appropriate across the full transaction value range**. The correct approach is a threshold schedule that varies with transaction characteristics.
 
### Feature importance: which signals drive fraud detection
 
The XGBoost model's feature importance (gain-based) reveals that fraud prediction is dominated by a small subset of features:
 
| Rank | Feature | Interpretation |
|---|---|---|
| 1 | V14 | Strongest individual fraud signal; likely captures an interaction involving transaction velocity or merchant-account pattern anomaly |
| 2 | V4 | Positively associated with fraud in EDA; may reflect behavioral deviation from account baseline |
| 3 | V12 | High gain; likely captures a cross-dimensional interaction not separable in lower-dimensional projections |
| 4 | Amount | Fraud transactions cluster at lower amounts — consistent with card testing behavior |
| 5 | V11 | Moderate importance; directionally consistent with fraud class separation observed in EDA |
 
The anonymization of V1–V28 limits the interpretability of these findings in absolute terms. In a production deployment with access to raw features, the top contributors would be translated into plain-language risk factors: "unusual transaction amount for account," "merchant category inconsistent with account history," and so on. This translation step is a product requirement, not a modeling exercise.
 
---
 
## From Model Output to Operational Decision
 
The model produces a continuous risk score between 0 and 1. Converting this score into an operational decision requires three additional design choices that sit outside the model:
 
### Threshold calibration
 
The classification threshold determines the Recall–Precision trade-off. Lowering the threshold catches more fraud (higher Recall) but also generates more false positives (lower Precision). The Precision-Recall curve from this model provides a menu of operating points — the business chooses a point based on its loss function.
 
A typical approach:
- Set a primary threshold targeting a Recall of ~90% as the baseline operating point
- Maintain a secondary lower threshold that routes borderline cases to human review rather than automatically blocking
- Adjust thresholds dynamically in response to fraud rate changes (e.g., following a data breach event, temporarily lower the threshold to increase sensitivity)
 
### Explainability for declined transactions
 
In most markets, financial institutions are legally required to provide a reason code when a transaction is declined. The XGBoost model supports this through SHAP (SHapley Additive exPlanations) values, which decompose the model's score into per-feature contributions for each individual transaction. A transaction declined due to high V14 and unusual Amount would generate a reason statement such as: *"This transaction was flagged due to an unusual transaction pattern inconsistent with account history."*
 
This requirement eliminates neural networks as viable alternatives in regulated deployment contexts, regardless of benchmark performance gains.
 
### Model drift monitoring
 
Fraud attackers adapt to detection systems. A model achieving 91.5% Recall at deployment may degrade significantly within months. Production requirements for monitoring include:
 
- **Weekly Recall tracking** on labeled samples from the dispute feedback loop
- **Alert thresholds** triggering retraining pipelines when Recall drops below a defined floor (typically 5–10 percentage points below baseline)
- **Champion–Challenger framework**: new model versions are A/B tested against the production model on live traffic before full rollout, preventing regressions from entering production silently
 
---
 
## Limitations & Open Questions
 
**Feature opacity.** The PCA anonymization of V1–V28 prevents direct interpretation of what behavioral signals drive the model. This is a dataset constraint rather than a modeling choice, but it limits the project's ability to generate actionable feature-level insights that would transfer to real deployment.
 
**Temporal validation.** The train/test split in this project is random rather than time-ordered. In practice, a fraud detection model should be validated on future data relative to its training period — otherwise, the evaluation does not reflect the model's actual generalization challenge. A proper validation would use the first ~80% of the two-day window for training and hold out the final ~20% for testing.
 
**Static threshold.** This project evaluates models at a fixed threshold. Production systems require threshold scheduling across transaction segments and dynamic adjustment in response to changing fraud rates.
 
**No network features.** Transaction-level features capture individual anomalies but miss network-level patterns — shared devices, linked accounts, coordinated fraud rings. Graph-based features (e.g., shared IP, device clustering, velocity across linked accounts) would meaningfully improve detection of organized fraud, but require cross-transaction data that is not available in this single-table dataset.
 
**Feedback loop not modeled.** The dispute resolution pipeline — where false positives and false negatives are relabeled and fed back into training — is described architecturally but not implemented. In a real system, the quality and latency of this feedback loop is a primary determinant of how quickly the model recovers from emerging fraud patterns.
 
---

## Repository Structure

```
credit-card-fraud-detection/
├── data/
│   ├── raw/         
│   └── processed/    
│
├── notebooks/
│   ├── 01_EDA.ipynb                  # Exploratory data analysis
│   ├── 02_preprocessing.ipynb        # Feature engineering + SMOTE
│   ├── 03_modeling.ipynb             # Model training + comparison
│   └── 04_evaluation.ipynb           # Results + business interpretation
│
├── src/
│   ├── preprocessing.py              # Data cleaning + scaling utilities
│   ├── models.py                     # Model training wrappers
│   └── evaluation.py                 # Metrics + visualization functions
│
├── reports/
│   └── figures/                      # Saved plots (PR curves, confusion matrix, etc.)
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Jihe377/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in `data/raw/`.

**4. Run notebooks in order**
```bash
jupyter notebook notebooks/01_EDA.ipynb
```

**Requirements**
```
python>=3.9
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
imbalanced-learn>=0.11
matplotlib>=3.7
seaborn>=0.12
jupyter>=1.0
```

---

## References

- Dal Pozzolo, A. et al. (2015). *Calibrating Probability with Undersampling for Unbalanced Classification.* IEEE SSCI.
- Bahnsen, A.C. et al. (2016). *Feature Engineering Strategies for Credit Card Fraud Detection.* Expert Systems with Applications.
- [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/) — Le Borgne et al., ULB Machine Learning Group

---

## Author

**Danyan Gu**

[![GitHub](https://img.shields.io/badge/GitHub-Jihe377-181717?logo=github)](https://github.com/Jihe377)
[![Email](https://img.shields.io/badge/Email-diana.gu.317%40gmail.com-D14836?logo=gmail&logoColor=white)](mailto:diana.gu.317@gmail.com)

---

*MIT License — Data sourced from ULB Machine Learning Group via Kaggle*
