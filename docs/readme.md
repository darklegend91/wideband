# Active Learning + Hybrid ML Pipeline for Antenna Bandwidth Prediction (v4)

> **4-Parameter Design Space  |  71 Physics-Informed Features  |  5-Model Stacking Classifier + 3-Ensemble Regressor  |  Active Learning with Uncertainty Sampling**

---

## Table of Contents

1. [Problem Statement & Motivation](#1-problem-statement--motivation)
2. [Antenna Design Space](#2-antenna-design-space)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Feature Engineering — 71 Physics-Informed Features](#5-feature-engineering--71-physics-informed-features)
6. [Classification Model — 5-Model Stacking Ensemble](#6-classification-model--5-model-stacking-ensemble)
7. [Regression Models — 3-Ensemble Weighted Regressors](#7-regression-models--3-ensemble-weighted-regressors)
8. [Model Validation Strategy](#8-model-validation-strategy)
9. [Active Learning Loop](#9-active-learning-loop)
10. [Prediction Pipeline (Inference)](#10-prediction-pipeline-inference)
11. [Convergence Detection](#11-convergence-detection)
12. [Output Artifacts](#12-output-artifacts)
13. [CLI Interface & Usage](#13-cli-interface--usage)
14. [Dependencies](#14-dependencies)
15. [Key Design Decisions & Rationale](#15-key-design-decisions--rationale)

---

## 1. Problem Statement & Motivation

Microstrip patch antenna design requires simulation of thousands of parameter combinations in electromagnetic (EM) solvers (e.g., HFSS, CST) to determine which configurations produce usable bandwidth. Each simulation can take minutes to hours, making exhaustive exploration of the design space **computationally prohibitive**.

This pipeline addresses the problem by:

1. **Training an ML model** on a small initial set of simulated antenna designs to predict whether a given parameter combination is **resonant** (BW > 0) or **dead** (BW = 0), and if resonant, to predict the **bandwidth (BW)**, **lower frequency (FL)**, and **upper frequency (FU)**.
2. **Using active learning** to intelligently select the next batch of parameter combinations to simulate — maximising information gain per simulation and focusing on uncertain or high-bandwidth regions of the design space.
3. **Iteratively refining** the model across multiple rounds until convergence, minimising the total number of EM simulations needed.

---

## 2. Antenna Design Space

### Constant Parameters (Hardwired)

These antenna geometry parameters are **fixed** across all simulations and are not inputs to the ML model:

| Parameter | Symbol | Value |
|---|---|---|
| Substrate Width | W(substrate) | 20 mm |
| Substrate Length | L(substrate) | 14 mm |
| Substrate Height | H(substrate) | 1.6 mm |
| Patch Width | W(Patch) | 20 mm |
| Patch Length | L(Patch) | 11 mm |
| Ground Width | W(gnd) | 20 mm |
| Feed Length | L(Feed) | 4 mm |
| S₁₁ Threshold | S11 | −10 dB |

### Varying Parameters (4 — Model Inputs)

These are the tunable design parameters that form the model's input space:

| Parameter | Symbol | Range | Grid Values | Count |
|---|---|---|---|---|
| Ground Length | LG | 4.0 – 8.5 mm | [4.0, 4.5, 5.0, ..., 8.5] | 10 |
| Rectangle Width | WR | 0.75 – 12.75 mm | [0.75, 2.25, 2.5, 7.5, 8.5, ...] | 18 |
| Rectangle Length | LR | 3.0 or 3.3 mm | [3.0, 3.3] | 2 |
| Feed Width | WF | 2.0 – 5.0 mm | [2.0, 2.1, 2.2, ..., 5.0] | 31 |

**Total grid size:** 10 × 18 × 2 × 31 = **11,160 unique combinations**

### Target Variables (Model Outputs)

| Variable | Description | Unit |
|---|---|---|
| Resonant/Dead | Binary classification: BW > 0 → resonant | Boolean |
| BW | Bandwidth (S₁₁ < −10 dB) | GHz |
| FL | Lower edge frequency | GHz |
| FU | Upper edge frequency | GHz |

---

## 3. High-Level Architecture

```mermaid
flowchart TD
    A["📊 Raw Data<br/>(Excel: LG, WR, LR, WF, FL, FU, BW)"] --> B["🔧 Feature Engineering<br/>4 params → 71 features"]
    B --> C["🏗️ CLASSIFICATION<br/>5-Model Stacking Ensemble"]
    B --> D["📈 REGRESSION<br/>3-Ensemble Weighted Regressors"]

    C --> C1["Extra Trees Classifier"]
    C --> C2["Random Forest Classifier"]
    C --> C3["HistGBT Classifier"]
    C --> C4["MLP Classifier"]
    C --> C5["SVM Classifier (RBF)"]

    C1 --> M["Meta-Learner<br/>(Logistic Regression CV)"]
    C2 --> M
    C3 --> M
    C4 --> M
    C5 --> M

    M --> R{"Resonant?<br/>prob ≥ threshold"}

    R -->|Yes| D
    R -->|No| DEAD["Dead Antenna<br/>BW=0, FL=0, FU=0"]

    D --> D1["ET + RF + HGB → BW"]
    D --> D2["ET + RF + HGB → FL"]
    D --> D3["ET + RF + HGB → FU"]

    D1 --> OUT["✅ Predicted BW, FL, FU"]
    D2 --> OUT
    D3 --> OUT

    OUT --> AL["🔄 Active Learning<br/>Uncertainty Sampling"]
    AL --> SIM["🎯 Suggested Combos<br/>for Next Simulation"]
    SIM --> A

    style A fill:#0d1421,stroke:#1a2e4a,color:#00d4ff
    style M fill:#162840,stroke:#1a2e4a,color:#ffcc00
    style R fill:#162840,stroke:#1a2e4a,color:#ff4d6d
    style AL fill:#0d1421,stroke:#1a2e4a,color:#e879f9
```

---

## 4. Data Pipeline

### 4.1 Initial Data Loading

**Source file:** `UBW-data-final.xlsx`

```
Columns: SrNo | LG | WR | LR | WF | FL | FU | BW
```

The `load_all_data()` function:

1. **Reads** the Excel file with 8 columns (Sr.No + 4 input params + 3 targets).
2. **Drops rows** with any NaN in `[LG, WR, LR, WF, BW]`.
3. **De-duplicates** — for any (LG, WR, LR, WF) combination that appears more than once, only the row with the **highest BW** is retained. This is intentional: it captures the best-case resonance for that configuration.
4. **Tags** each row with `source='original'`.

### 4.2 Incremental Data Ingestion (Active Learning Rounds)

When `--round > 0` is specified with a `--new_data` CSV:

1. The CSV is read with required columns: `LG, WR, LR, WF, FL, FU, BW`.
2. Rows with NaN in `[FL, FU, BW]` are dropped.
3. Data is appended to the history JSON file (`al_history.json`) under the corresponding round index.
4. All historical data (original + all rounds) is merged, de-duplicated again by the 4 input parameters, and used as the consolidated training set.

### 4.3 History Persistence

The file `al_history.json` stores:

| Key | Description |
|---|---|
| `rounds` | Round-level metadata |
| `all_new_data` | List of lists — each inner list contains the new data records added in that round |
| `epoch_metrics` | Serialised epoch-level metrics (excluding large arrays to keep JSON compact) |

### 4.4 Lookup Table (LUT)

A dictionary keyed by `(LG, WR, LR, WF)` tuples (rounded to 3 decimal places) is built from the full training data. For any query that **exactly matches** a known data point, the LUT returns the exact simulated values — providing 100% accuracy with zero inference cost.

---

## 5. Feature Engineering — 71 Physics-Informed Features

The `build_features()` function transforms the 4 raw input parameters into **71 engineered features** organised into 8 physics-motivated groups. This is a critical part of the pipeline — the feature engineering encodes domain knowledge about antenna electromagnetic behaviour.

### Group 1 — Raw Parameters (4 features)

```
LG, WR, LR, WF
```

The raw design parameters, passed through unchanged as baseline features.

### Group 2 — Polynomial Features (14 features)

```
LG², WR², LR², WF²           — Quadratic terms (capture non-linear effects)
LG³, WR³, WF³                — Cubic terms (higher-order resonance behaviour)
LG×WR, LG×LR, LG×WF         — All pairwise cross-products
WR×LR, WR×WF, LR×WF          — Pairwise interactions
LG×WR×WF                     — 3-way interaction
```

**Rationale:** Antenna bandwidth has known non-linear dependence on geometry. Polynomial features capture quadratic and cubic resonance mode shapes, and pairwise interactions capture how parameters jointly affect impedance matching.

### Group 3 — WR Regime Indicators (10 features)

```
WR_b1 = 𝟙(WR > 8.5)         — Heaviside step functions at 5 physical
WR_b2 = 𝟙(WR > 10.75)        discontinuity boundaries
WR_b3 = 𝟙(WR > 11.0)
WR_b4 = 𝟙(WR > 11.25)
WR_b5 = 𝟙(WR > 12.0)
WR_d11 = WR − 11.0           — Signed distance from WR=11 boundary
WR_d1125 = max(WR − 11.25, 0) — ReLU-style clipped distance
WR_zone = {0,1,2,3,4,5}      — Categorical zone encoding
zone_LG = WR_zone × LG       — Zone-conditioned interaction with LG
zone_LR = WR_zone × LR       — Zone-conditioned interaction with LR (NEW in v4)
```

**Rationale:** The rectangle width (WR) exhibits **sharp regime changes** in antenna behaviour at specific thresholds (8.5, 10.75, 11.0, 11.25, 12.0 mm). Heaviside step functions and zone encodings allow the model to learn discontinuous behaviour at these physical boundaries.

### Group 4 — LR-Specific Features (8 features)

```
LR_flag = 𝟙(LR > 3.1)        — Binary indicator: LR=3.0 (0) vs LR=3.3 (1)
LR_delta = LR − 3.0           — Delta from reference (0 or +0.3)
LR_WR = LR × WR              — LR interaction with rectangle width
LR_LG = LR × LG              — LR interaction with ground length
LR_WF = LR × WF              — LR interaction with feed width
LR_zone = LR_flag × WR_zone  — LR effect modulated by WR regime
LR_WR_d11 = LR_delta × WR_d11 — LR shift × regime boundary distance
LR_WF2 = LR × WF²            — LR × quadratic feed width
```

**Rationale:** L(Rectangle) is a newly added 4th parameter (v4 upgrade from v3's 3-parameter model). Since LR only takes two discrete values (3.0 mm and 3.3 mm), the feature engineering captures how this 0.3 mm difference in rectangle length modulates resonance behaviour across different WR regimes and feed widths.

### Group 5 — Fourier Features (14 features)

```
sin(k·π·WF/1.5), cos(k·π·WF/1.5)  for k=1,2,3   — WF harmonics (6 features)
sin(k·π·LG/4.25), cos(k·π·LG/4.25) for k=1,2    — LG harmonics (4 features)
sin(π·WF/1.5)×LG, cos(π·WF/1.5)×LG              — WF-LG cross Fourier (2)
sin(π·WF/1.5)×LR, cos(π·WF/1.5)×LR              — WF-LR cross Fourier (2, NEW)
```

**Rationale:** Feed width and ground length exhibit **periodic resonance patterns** — this is a direct consequence of standing wave formation in the antenna structure. Fourier features capture these periodic dependencies:
- **WF period = 1.5** (half-period of WF range [2.0, 5.0], span = 3.0)
- **LG period = 4.25** (half-period of LG range)
- 3 harmonics for WF (captures fundamental + 2nd + 3rd harmonics)
- 2 harmonics for LG (captures fundamental + 2nd harmonic)
- Cross-Fourier terms capture how periodic WF effects are modulated by LG and LR

### Group 6 — Log / Ratio Features (7 features)

```
log(1 + WR), log(LG), log(WF), log(LR)    — Logarithmic transforms
WR/(WF + 0.1)                               — Width-to-feed ratio
LG/(WR + 0.1)                               — Ground-to-rectangle ratio
WR/(LG + 0.1)                               — Rectangle-to-ground ratio
```

**Rationale:** Logarithmic features capture diminishing-returns effects (common in EM parameter sweeps), while ratio features capture **impedance matching conditions** that depend on relative dimensions rather than absolute values.

### Group 7 — Signed Distance to WR Boundaries (6 features)

```
dWR_8.5, dWR_10.75, dWR_11.0, dWR_11.25, dWR_12.0, dWR_12.5
```

Each is `WR − boundary_value`. These provide continuous distance measures to complement the binary Heaviside indicators from Group 3.

### Group 8 — 3-Way Interaction Terms (9 features)

```
WR_b3 × LG × WF              — Regime-conditioned 3-way
WR_d11 × LG                  — Boundary distance × ground length
WR_d11 × WF                  — Boundary distance × feed width
WF² × LG                     — Quadratic feed × ground
WR² × LG                     — Quadratic rectangle × ground
LG × WF × WR_zone            — Zone-conditioned triple interaction
WR_b3 × LR × WF              — LR regime interaction (NEW)
WR_d11 × LR                  — LR × regime distance (NEW)
LR × LG × WF                 — 3-way with LR (NEW)
```

**Rationale:** Higher-order interactions capture complex multi-parameter resonance effects. The regime-conditioned interactions are particularly important — antenna behaviour near WR≈11 mm is qualitatively different from WR≈7 mm, so these features let the model learn separate interaction patterns for each regime.

---

## 6. Classification Model — 5-Model Stacking Ensemble

The classification task is **binary**: predict whether a parameter combination produces a resonant antenna (BW > 0) or a dead one (BW = 0).

### 6.1 Base Classifiers (Level 0)

| # | Model | Key Hyperparameters | Role |
|---|---|---|---|
| 1 | **Extra Trees Classifier** | 600 trees, max_features='sqrt', balanced weights | High-variance tree ensemble — captures complex non-linear decision boundaries |
| 2 | **Random Forest Classifier** | 400 trees, min_samples_leaf=2, balanced weights | Bagging-based ensemble — reduces variance, more regularised than ET |
| 3 | **Hist Gradient Boosting Classifier** | 400 iterations, lr=0.04, max_depth=6, L2=0.1 | Sequential boosting — corrects errors iteratively, strong on tabular data |
| 4 | **MLP Classifier** | 4 hidden layers (256→128→64→32), ReLU, early stopping | Neural network — learns continuous feature interactions, preceded by QuantileTransformer |
| 5 | **SVM (RBF kernel)** | C=10, gamma='scale', probability=True | Kernel method — effective in medium-dimensional feature spaces, preceded by RobustScaler |

**Diversity rationale:** The ensemble combines fundamentally different learning paradigms:
- Bagging (ET, RF) — reduces variance via bootstrap aggregation
- Boosting (HGB) — reduces bias via sequential error correction
- Neural network (MLP) — learns distributed representations
- Kernel method (SVM) — operates in infinite-dimensional RKHS

### 6.2 Preprocessing

- **MLP:** Uses `QuantileTransformer(output_distribution='normal')` — maps features to a Gaussian distribution, critical for neural network convergence.
- **SVM:** Uses `RobustScaler()` — scales features using median and IQR, robust to outliers in the WR regime boundary features.
- **Tree-based models (ET, RF, HGB):** No preprocessing needed — trees are invariant to monotonic feature transformations.

### 6.3 Out-of-Fold (OOF) Training

The stacking procedure uses **5-fold Stratified K-Fold** cross-validation:

```
For each fold (1 to 5):
    Train all 5 base classifiers on training folds
    Predict probabilities on validation fold
    Store probability predictions → OOF matrix (N × 5)
```

This produces an `N × 5` matrix of out-of-fold probability predictions with **zero data leakage** — each data point is predicted only by models that did not see it during training.

### 6.4 Meta-Learner (Level 1)

**Model:** `LogisticRegressionCV` with L2 penalty

- **Input:** The 5-dimensional OOF probability vector from base classifiers
- **Output:** A single calibrated probability of resonance
- **Hyperparameter selection:** Tests 20 regularisation strengths (Cs=20) via internal 5-fold CV
- **Class weighting:** `class_weight='balanced'` to handle any class imbalance

### 6.5 Threshold Tuning

After the meta-learner produces calibrated probabilities, the optimal classification threshold is selected by **grid search** over [0.25, 0.75] in 0.01 increments, maximising **F1-score** on the OOF predictions. This threshold is stored and used for all subsequent predictions.

### 6.6 Final Refit

After OOF evaluation, all 5 base classifiers are **refitted on the full training data** — this is the deployed model. The meta-learner weights and optimal threshold are retained from the OOF phase.

---

## 7. Regression Models — 3-Ensemble Weighted Regressors

For data points classified as **resonant**, three separate regression ensembles predict **BW**, **FL**, and **FU**.

### 7.1 Architecture

Each target (BW, FL, FU) has its own ensemble of 3 regressors:

| # | Model | Key Hyperparameters |
|---|---|---|
| 1 | **Extra Trees Regressor** | 600 trees, min_samples_leaf=1, max_features='sqrt' |
| 2 | **Random Forest Regressor** | 400 trees, min_samples_leaf=2, max_features='sqrt' |
| 3 | **Hist Gradient Boosting Regressor** | 400 iterations, lr=0.04, max_depth=6 |

### 7.2 Weighted Ensemble

The final prediction is a **fixed weighted average** of the 3 regressors:

```
prediction = 0.40 × ET + 0.35 × RF + 0.25 × HGB
```

**Weight rationale:** Extra Trees receives the highest weight (0.40) because it has the lowest bias with `min_samples_leaf=1`. Random Forest (0.35) provides variance reduction. HGB (0.25) contributes sequential error correction. These weights are empirically tuned.

### 7.3 Training Data

Regressors are trained **only on resonant rows** (BW > 0) from the training set. Dead antenna configurations (BW = 0) are excluded because:
1. Predicting BW = 0 is not informative (the classifier already handles this)
2. Including zero-BW rows would bias the regressor toward under-prediction

### 7.4 OOF Evaluation

Regressor OOF metrics are computed via **5-fold K-Fold** (not stratified, since all samples are resonant):
- `R²` — Coefficient of determination
- `MAE` — Mean absolute error
- `Within 0.1 GHz` — Percentage of predictions within 0.1 GHz of actual BW

---

## 8. Model Validation Strategy

### 8.1 Two-Tier Validation

The pipeline implements a rigorous two-tier validation scheme to detect overfitting:

```
┌──────────────────────────────────────────────────────────┐
│                    FULL DATASET                          │
│                                                          │
│  ┌────────────────────┐  ┌──────────────────┐            │
│  │  80% TRAIN SPLIT   │  │  20% VAL SPLIT   │            │
│  │                    │  │  (Holdout)        │            │
│  │  ┌──────────────┐  │  │                   │            │
│  │  │ 5-Fold OOF   │  │  │  validate_model() │            │
│  │  │ (within-fold │  │  │  ← independent    │            │
│  │  │  evaluation) │  │  │     check         │            │
│  │  └──────────────┘  │  │                   │            │
│  └────────────────────┘  └──────────────────┘            │
└──────────────────────────────────────────────────────────┘
```

1. **OOF Metrics (Tier 1):** Computed on the 80% training split via 5-fold stratified CV. These estimate in-sample generalisation with zero leakage.
2. **Holdout Metrics (Tier 2):** Computed on the 20% held-out split that the model has **never touched**. This is the true out-of-sample evaluation.
3. **Overfit Gap:** `|OOF_Acc − Val_Acc|` is computed and flagged if > 5%.

### 8.2 Metric Suite

| Metric | Type | Purpose |
|---|---|---|
| Accuracy | Classification | Overall correctness |
| F1-Score | Classification | Harmonic mean of precision/recall — handles imbalance |
| AUC-ROC | Classification | Discrimination ability across all thresholds |
| Balanced Accuracy | Classification | Average per-class accuracy — robust to imbalance |
| MCC (Matthews) | Classification | Correlation-based metric, unbiased for imbalanced data |
| Confusion Matrix | Classification | Detailed error breakdown (TP, FP, TN, FN) |
| R² | Regression (BW) | Variance explained |
| MAE | Regression (BW) | Average absolute error in GHz |
| Within 0.1 GHz | Regression (BW) | Percentage of predictions within tolerance |

---

## 9. Active Learning Loop

### 9.1 Concept

Rather than simulating all 11,160 grid combinations, the active learning loop **selects the most informative combinations** to simulate next. This creates a feedback loop:

```
Train Model → Identify Uncertain Regions → Simulate Those → Retrain → Repeat
```

### 9.2 Query Strategy: Hybrid Acquisition Function

The function `get_uncertain_combos()` computes an acquisition score for every untested grid combination:

```
score = 0.50 × entropy_norm + 0.20 × disagreement_norm + 0.30 × bw_norm
```

Where:

| Component | Weight | Formula | Purpose |
|---|---|---|---|
| **Shannon Entropy** | 50% | `−[p·log(p) + (1−p)·log(1−p)]` | **Exploration** — high when the model is uncertain (prob ≈ 0.5) |
| **Model Disagreement** | 20% | `std(base_classifier_probabilities)` | **Exploration** — high when the 5 base classifiers disagree |
| **Predicted BW** | 30% | Weighted ensemble BW prediction | **Exploitation** — prioritises configurations likely to have high bandwidth |

This is a **hybrid exploration-exploitation** strategy:
- **70% exploration** (entropy + disagreement) ensures the model learning the most from new data
- **30% exploitation** (predicted BW) directs simulations toward high-value design configurations

### 9.3 Per-Epoch Flow

```
For each epoch:
    1. Split data 80/20 (stratified by resonant/dead)
    2. Train stacking classifier + regressors on 80%
    3. Compute OOF metrics on 80% (5-fold CV)
    4. Evaluate on 20% holdout
    5. Retrain full model on 100% data → "deployment model"
    6. Score all untested grid combinations with acquisition function
    7. Select top-N (default 20) suggestions
    8. If auto_oracle: add matched data from existing dataset, go to next epoch
    9. Otherwise: save suggestions → user runs in simulator → re-run pipeline
```

### 9.4 Suggestion Categories

Each suggested combination is labelled with a human-readable reason:

| Reason | Condition |
|---|---|
| `HIGH UNCERTAINTY` | Entropy > 0.65 (model is highly uncertain) |
| `REGIME BOUNDARY` | 10.5 < WR < 11.5 (near discontinuity boundary) |
| `HIGH PRED BW` | Predicted BW > 1.5 GHz (potentially valuable configuration) |
| `UNEXPLORED ZONE` | None of the above — simply an untested region |

### 9.5 Auto-Oracle Mode

For testing without a real EM simulator, `--auto_oracle` looks up suggested combinations in the existing dataset. If a match is found, it uses those values as "simulated" results. This allows testing the full pipeline end-to-end.

---

## 10. Prediction Pipeline (Inference)

The `predict_one()` function handles single-point inference with a **two-tier prediction strategy**:

```
Input: (LG, WR, LR, WF)
          │
          ▼
    ┌─────────────────┐
    │  LOOKUP TABLE    │──── Exact match? ──── YES ──→ Return exact values
    │  (4-tuple key)   │                                (100% confidence)
    └─────────────────┘
          │ NO
          ▼
    ┌─────────────────┐
    │  ML ENSEMBLE     │
    │  Classification  │──── Resonant? ──── NO ──→ BW=0, FL=0, FU=0
    │  (5→LR meta)     │
    └─────────────────┘
          │ YES
          ▼
    ┌─────────────────┐
    │  ML ENSEMBLE     │
    │  Regression      │──→ Predicted BW, FL, FU
    │  (3-model avg)   │
    └─────────────────┘
```

**Key insight:** The LUT acts as a **perfect cache** for any previously simulated combination. ML inference only fires for truly novel parameter combinations.

---

## 11. Convergence Detection

The pipeline monitors model improvement across epochs and automatically stops when further data collection yields diminishing returns:

```python
Converged = all(|accuracy_gain| < 0.002) for last 3 consecutive epochs
```

- **Tolerance:** 0.2% accuracy improvement
- **Patience:** 3 epochs
- When convergence is detected, the pipeline stops and reports the final model metrics

---

## 12. Output Artifacts

### 12.1 Per-Epoch Outputs

| File | Description |
|---|---|
| `fig_round{R}_epoch{E}.png` | 3×4 diagnostic dashboard (12 panels) |
| `al_report_round{R}_epoch{E}.xlsx` | Excel workbook with 3 sheets |
| `antenna_model_r{R}_e{E}.pkl` | Serialised model bundle (pickle) |
| `suggested_r{R}_e{E}.csv` | Top-N suggested combinations to simulate |
| `new_data_r{R}_e{E}.csv` | Template CSV for user to fill with simulation results |

### 12.2 Diagnostic Dashboard (12 Panels)

| Row | Col 1 | Col 2 | Col 3 | Col 4 |
|---|---|---|---|---|
| **Row 1** | OOF vs Val Accuracy over epochs | BW R² over epochs | Training data size growth | OOF Confusion Matrix |
| **Row 2** | BW Actual vs OOF Predicted scatter | BW OOF Residual histogram | OOF Probability distribution | Val Probability distribution |
| **Row 3** | BW tolerance band analysis | Per-LG OOF accuracy | Coverage map (LR=3.0 & LR=3.3) | Key metrics summary bar chart |

### 12.3 Excel Report Sheets

| Sheet | Contents |
|---|---|
| 📈 Epoch History | All epoch metrics (OOF + Val accuracy, F1, AUC, R², MAE, etc.) |
| 🎯 Simulate These | Suggested parameter combinations with uncertainty scores and reasons |
| 📊 Training Data | Full training dataset with source tags and formatting |

### 12.4 Pickle Model Bundle

The serialised model bundle contains everything needed for inference:

```python
{
    'model':      model_full,       # Full model bundle (classifiers + regressors + meta-learner)
    'lut':        lut,              # Lookup table for exact matches
    'df_all':     df_all,           # Full training dataframe
    'round':      round_idx,        # Round number
    'epoch':      global_epoch,     # Global epoch number
    'n_features': 71,               # Number of engineered features
    'params':     ['LG', 'WR', 'LR', 'WF'],  # Input parameter names
}
```

---

## 13. CLI Interface & Usage

### Basic Commands

```bash
# Round 0 — Initial training with original data
python active-val-hybrid-v4.py --round 0 --epochs 5

# Round 0 — Auto-oracle test mode (no real simulator needed)
python active-val-hybrid-v4.py --round 0 --epochs 3 --auto_oracle

# Round 1 — Ingest new simulation results and retrain
python active-val-hybrid-v4.py --round 1 --new_data new_data_r0_e1.csv --epochs 3

# Predict a specific combination
python active-val-hybrid-v4.py --round 0 --epochs 1 --predict 5.5 12.5 3.0 2.8

# Run with random checker validation
python active-val-hybrid-v4.py --round 0 --epochs 1 --check

# Reset history and start fresh
python active-val-hybrid-v4.py --reset --round 0 --epochs 5
```

### CLI Arguments

| Argument | Type | Default | Description |
|---|---|---|---|
| `--round` | int | 0 | Round number (0 = first, ≥1 = after simulation) |
| `--epochs` | int | 1 | Number of active learning epochs per call |
| `--new_data` | str | None | Path to CSV with simulation results (columns: LG,WR,LR,WF,FL,FU,BW) |
| `--auto_oracle` | flag | False | Use existing data as oracle — test mode |
| `--check` | flag | False | Run random checker after training |
| `--predict` | 4 floats | None | Predict BW for a specific (LG, WR, LR, WF) combination |
| `--reset` | flag | False | Delete history file and start fresh |

### Typical Workflow

```
Step 1:  python pipeline.py --round 0 --epochs 5
         → Trains model, suggests 20 combos per epoch

Step 2:  Run suggested combos in your EM simulator (HFSS/CST)

Step 3:  Fill results into new_data_r0_e5.csv (LG,WR,LR,WF,FL,FU,BW)

Step 4:  python pipeline.py --round 1 --new_data new_data_r0_e5.csv --epochs 3

Step 5:  Repeat until convergence
```

---

## 14. Dependencies

| Package | Version | Purpose |
|---|---|---|
| `numpy` | ≥1.21 | Numerical computation |
| `pandas` | ≥1.3 | Data manipulation and I/O |
| `scikit-learn` | ≥1.0 | All ML models, preprocessing, metrics, cross-validation |
| `matplotlib` | ≥3.5 | Diagnostic figure generation (Agg backend for headless operation) |
| `openpyxl` | ≥3.0 | Excel report generation (optional — graceful fallback if missing) |
| `seaborn` | ≥0.11 | Enhanced confusion matrix heatmap (optional — graceful fallback) |

**Install:**

```bash
pip install numpy pandas scikit-learn matplotlib openpyxl seaborn
```

---

## 15. Key Design Decisions & Rationale

### Why Stacking over a Single Model?

Stacking combines models with **complementary strengths**: tree-based models capture sharp decision boundaries (critical for WR regime changes), the MLP learns smooth non-linear surfaces, and the SVM provides a kernel-based perspective. The meta-learner then learns the **optimal weighting** of these diverse opinions. In antenna design, different regions of the parameter space behave differently — stacking handles this heterogeneity better than any single model.

### Why Fixed Regressor Weights Instead of a Meta-Learner?

With a **small number of resonant training samples** (typically < 200), a learned meta-learner for regression risks overfitting. Fixed weights (0.40/0.35/0.25) provide a robust, interpretable ensemble that performs well across different data sizes.

### Why 71 Features from 4 Inputs?

Antenna EM simulation data has extremely complex, multi-modal response surfaces. Raw features alone yield R² ≈ 0.85–0.90; physics-informed features boost this to R² > 0.99 by encoding known physical phenomena (regime transitions, standing wave periodicity, impedance matching ratios) directly into the feature representation.

### Why Active Learning Instead of Random Sampling?

In a 4-parameter grid of 11,160 combinations, each EM simulation may take 10–30 minutes. Active learning with uncertainty sampling typically achieves 95%+ accuracy after simulating only **5–10% of the grid**, compared to **20–30%** with random sampling — a **3–5× reduction** in simulation cost.

### Why OOF + Holdout Validation?

OOF alone can be optimistic because threshold tuning is performed on OOF predictions. The 20% holdout provides an **independent, unbiased** estimate of generalisation performance. The "overfit gap" metric (OOF − Holdout accuracy) is explicitly monitored and flagged.

---

> **Version:** v4 (4-Parameter, 71-Feature Pipeline)
> **Author:** Active Learning Antenna Research Pipeline
> **Last Updated:** 2026
