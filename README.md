# Credit Card Fraud Detection
### A Risk Control Product Perspective

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

## Table of Contents

- [Business Context](#-business-context)
- [Dataset](#-dataset)
- [Key Business Decisions](#-key-business-decisions)
- [Technical Pipeline](#-technical-pipeline)
- [Results](#-results--business-interpretation)
- [Production Considerations](#-production-considerations)
- [Connection to Risk Control Experience](#-connection-to-risk-control-experience)
- [Repository Structure](#-repository-structure)
- [How to Run](#-how-to-run)

---

## Business Context

Credit card fraud detection is fundamentally a **resource allocation problem under uncertainty**. A payment system must make a `pass / flag / block` decision on every transaction within **100–300 milliseconds**, while minimizing two competing costs simultaneously:

| Error Type | Technical Term | Business Impact | Priority |
|---|---|---|---|
| Missed fraud | False Negative | Direct financial loss + user trust damage + compliance exposure | 🔴 **HIGH** |
| Blocked legit transaction | False Positive | User friction — recoverable via dispute process | 🟡 **MEDIUM** |

**Why this matters for model design:**

A naive model that classifies *every transaction as legitimate* achieves **99.83% accuracy** on this dataset — yet catches zero fraud. This illustrates why **Accuracy is a misleading metric in fraud detection**. The real optimization target is the Recall–Precision trade-off, and the threshold between them is a *business decision*, not a technical one.

> The classification threshold reflects the business's risk appetite. A lower threshold increases Recall (fewer missed frauds) at the cost of Precision (more false alarms). This is the same trade-off I encountered in content risk control at Meituan: over-blocking harms legitimate users; under-blocking harms platform integrity.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total transactions | 284,807 |
| Fraudulent transactions | 492 (0.172%) |
| Features | 28 PCA-anonymized features + `Amount` + `Time` |
| Class imbalance ratio | ~577:1 (legitimate : fraud) |
| Evaluation strategy | 5-fold stratified cross-validation |
| Primary metric | **Recall** on fraud class + AUC-PR |

**Why AUC-PR over AUC-ROC?**

AUC-ROC can be misleadingly optimistic on highly imbalanced datasets because it averages performance across all classification thresholds including regions irrelevant to the business problem. AUC-PR focuses specifically on the minority class (fraud), making it a more honest measure of model utility.

---

## Key Business Decisions

### Decision 1 — Why prioritize Recall over Accuracy

Standard accuracy is maximized by predicting all transactions as legitimate (99.83% accuracy, 0% fraud caught). The primary metric was changed to **Recall on the fraud class**, reflecting asymmetric cost structure:

- A **missed fraud** (False Negative): combined financial loss + dispute handling + regulatory exposure typically costs **10–50× more** than a false positive
- A **false positive**: user friction, recoverable via dispute/appeal process

This means the acceptable Recall–Precision trade-off is a business decision that depends on transaction value, fraud rate trend, and regulatory environment — not a technical hyperparameter to be grid-searched.

---

### Decision 2 — Why SMOTE over threshold adjustment alone

Two approaches address class imbalance from different angles:

| Approach | Mechanism | Trade-off |
|---|---|---|
| **Threshold adjustment** | Lower the decision boundary at inference time | Fast, but doesn't improve the model's *understanding* of fraud patterns |
| **SMOTE (used here)** | Generate synthetic fraud samples during training | Higher compute cost, but produces better generalization to unseen fraud patterns |

**In production, both are used together:**
- SMOTE (or similar) during training to improve base model quality
- Dynamic threshold tuning at inference to adjust the Recall–Precision balance in response to shifting fraud rates or changing business risk appetite

---

### Decision 3 — Why XGBoost over deep learning

Two factors drove this choice beyond raw benchmark performance:

**1. Interpretability is a hard requirement in risk control**

When a transaction is blocked, the system must explain *why* — to the user, the fraud analyst, and potentially a regulator. XGBoost's feature importance scores provide this auditability. A neural network cannot.

This mirrors the approach in content risk control: every rule update required documented justification. Interpretability is non-negotiable in regulated or high-stakes environments.

**2. Feature distribution robustness**

XGBoost handles missing values gracefully and is more robust to feature distribution drift — a practical advantage in production where fraud patterns shift continuously as attackers adapt.

---

## Technical Pipeline

```
Raw Data (284,807 transactions)
        │
        ▼
┌─────────────────────┐
│  Exploratory Analysis│  → Class imbalance visualization, feature correlation,
│                      │    Amount/Time distribution analysis
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Preprocessing       │  → StandardScaler on Amount + Time
│                      │    Train/test stratified split (80/20)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Class Imbalance     │  → SMOTE on training set only
│  Handling            │    (applied AFTER split to prevent data leakage)
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Model Training      │  → XGBoost, Random Forest,
│  & Comparison        │    Logistic Regression, SVM
│                      │    5-fold stratified cross-validation
└─────────────────────┘
        │
        ▼
┌─────────────────────┐
│  Evaluation          │  → Recall, Precision, F1, AUC-ROC, AUC-PR
│                      │    Precision-Recall curve, Confusion matrix
│                      │    Feature importance (XGBoost)
└─────────────────────┘
```

**Important: SMOTE applied after train/test split**

Applying SMOTE before splitting would cause data leakage — synthetic samples derived from test-set points would contaminate training, producing inflated metrics that don't reflect real-world generalization.

---

## Results & Business Interpretation

### Model comparison (5-fold cross-validation on test set)

| Model | Recall ↑ | Precision | F1 | AUC-ROC | AUC-PR |
|---|---|---|---|---|---|
| **XGBoost** | **91.5%** | 88.2% | 89.8% | 0.974 | 0.891 |
| Random Forest | 89.1% | 90.4% | 89.7% | 0.968 | 0.874 |
| Logistic Regression | 82.3% | 85.7% | 84.0% | 0.943 | 0.832 |
| SVM | 84.6% | 83.1% | 83.8% | 0.951 | 0.848 |

**XGBoost selected** on the basis of highest Recall (primary business metric) and best AUC-PR, with acceptable Precision trade-off.

---

### What does 91.5% Recall actually mean?

Out of every 100 fraudulent transactions:
- **~92 are correctly flagged** and can be blocked or reviewed
- **~8–9 are missed** and will result in financial loss

**Is this sufficient?** It depends entirely on business context:

| Scenario | Verdict | Reasoning |
|---|---|---|
| Low-value e-commerce (<$50) | Acceptable | Loss per missed fraud is bounded; false positives hurt conversion |
| Mid-value retail ($50–$500) | Marginal | Consider adding human review for flagged-but-uncertain cases |
| High-value wire transfers (>$500) | Insufficient | Each miss is costly; requires additional review layer |

This illustrates why **a single model threshold is never enough** for a real payment system — see Production Considerations below.

---

## Production Considerations

Moving from a batch ML experiment to a real-time fraud decisioning product requires addressing challenges that don't exist in a Kaggle notebook:

### 1. Latency constraint
Payment systems typically require **sub-300ms end-to-end decision time**. This constrains:
- Feature engineering (no cross-transaction lookups at inference time)
- Model complexity ceiling
- Infrastructure architecture (online serving, not batch scoring)

### 2. Three-tier decision architecture

A production fraud system is never a single model. Decisions are layered:

```
Transaction arrives
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 1: Rules Engine                  Latency: <10ms │
│  Hard-coded blocklist + velocity rules                │
│  → Handles: high-confidence known fraud               │
└──────────────────────────────────────────────────────┘
        │ (passes through)
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 2: ML Model                   Latency: 50–200ms │
│  XGBoost scoring + dynamic threshold                  │
│  → Handles: ambiguous transactions                    │
│    Output: PASS / BLOCK / REVIEW_QUEUE                │
└──────────────────────────────────────────────────────┘
        │ (uncertain cases)
        ▼
┌──────────────────────────────────────────────────────┐
│  TIER 3: Human Review Queue         Latency: min–hrs  │
│  Analyst-reviewed, prioritized by risk score          │
│  → Handles: low-confidence, high-value cases          │
└──────────────────────────────────────────────────────┘
```

### 3. Model drift monitoring

Fraud patterns evolve as attackers adapt. A model achieving 91.5% Recall today may drop to 75% within three months. Production requirements:
- Weekly monitoring of Recall and false positive rate on labeled samples
- Alert thresholds triggering automated retraining pipelines
- **Champion–Challenger framework**: A/B test new model versions against production before full rollout

### 4. Dispute & feedback loop

False positives (legitimate transactions blocked) generate user disputes. Dispute outcomes — *confirmed legitimate* or *confirmed fraud* — create labeled ground truth that feeds back into model retraining. Designing this feedback pipeline is a core product responsibility, not just an ML engineering task.

### 5. Explainability for customer-facing decisions

In most markets, financial institutions are required to provide a reason when declining a transaction. XGBoost feature importance enables statements like: *"This transaction was flagged due to unusual transaction amount combined with atypical merchant category for this account."* Neural networks cannot support this.

---

## Connection to Risk Control Experience

The core analytical skill in this project — **synthesizing weak correlated signals into a confident risk judgment** — transfers directly from content risk control to payment fraud detection.

At Meituan, I identified organized review fraud by combining signals across multiple dimensions (account age, device clustering, cross-platform behavioral gaps, WiFi association patterns). No single signal was conclusive; the judgment relied on multi-dimensional weak-signal fusion. This is structurally identical to multi-feature anomaly scoring in fraud detection, where fraud is rarely detectable from one feature alone.

**The key difference in payment fraud:**
- Latency is a hard constraint (milliseconds vs. hours available in content moderation)
- Each individual decision's financial impact is immediate and quantifiable
- The regulatory environment is more explicit

These differences change product design priorities — but not the underlying analytical framework.

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
[![Email](https://img.shields.io/badge/Email-gu.dany%40northeastern.edu-D14836?logo=gmail&logoColor=white)](mailto:gu.dany@northeastern.edu)

---

*MIT License — Data sourced from ULB Machine Learning Group via Kaggle*
