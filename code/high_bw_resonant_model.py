# Exported from active_val_high_bw_resonant_outputs/high_bw_resonant_model.ipynb
# Generated for the physics_guided_files bundle.


# %% [cell 0 - markdown]
# # High-Bandwidth Resonant Antenna Model
# 
# Goal: use the trained model to find designs with the **highest predicted bandwidth**, while keeping the model biased toward useful resonant antennas. This version uses only existing data and targets the practical frequency window around **4.5-9 GHz**.

# %% [cell 1 - markdown]
# ## 1. Imports and Configuration

# %% [cell 2 - code]
import os
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/wideband-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/wideband-cache")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, classification_report,
    r2_score, mean_absolute_error, mean_squared_error, log_loss
)

RANDOM_STATE = 42
DATA_PATH = Path("UBW-data-final.xlsx")
OUT_DIR = Path("active_val_high_bw_resonant_outputs")
OUT_DIR.mkdir(exist_ok=True)

TARGET_FL_MIN = 4.5
TARGET_FU_MAX = 9.0
MIN_RESONANT_RECALL = 0.93

LG_VALS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
WR_VALS = [0.75, 2.25, 2.5, 7.5, 8.5, 9.25, 9.5, 10.0, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75]
LR_VALS = [3.0, 3.3]
WF_VALS = [round(x, 1) for x in np.arange(2.0, 5.0 + 0.001, 0.1)]

print("Output:", OUT_DIR.resolve())

# %% [cell 3 - markdown]
# ## 2. Load Correct 4-Parameter Dataset

# %% [cell 4 - code]
def load_data():
    df = pd.read_excel(DATA_PATH, header=0, usecols=[0,1,2,3,4,5,6,7])
    df.columns = ["SrNo", "LG", "WR", "LR", "WF", "FL", "FU", "BW"]
    df = df.dropna(subset=["LG", "WR", "LR", "WF", "FL", "FU", "BW"]).copy()
    df[["LG", "WR", "LR", "WF", "FL", "FU", "BW"]] = df[["LG", "WR", "LR", "WF", "FL", "FU", "BW"]].astype(float)
    df = (df.sort_values("BW", ascending=False)
            .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
            .reset_index(drop=True))
    df["is_resonant"] = (df.BW > 0).astype(int)
    df["in_target_window"] = ((df.BW > 0) & (df.FL >= TARGET_FL_MIN) & (df.FU <= TARGET_FU_MAX)).astype(int)
    return df

df = load_data()
print("Rows:", len(df))
print("Resonant:", int(df.is_resonant.sum()))
print("Dead:", int((df.is_resonant == 0).sum()))
print("In 4.5-9 GHz target window:", int(df.in_target_window.sum()))
print("Max BW in data:", df.BW.max())
display(df[["LG", "WR", "LR", "WF", "FL", "FU", "BW", "is_resonant", "in_target_window"]].head(10))

# %% [cell 5 - markdown]
# ## 3. Feature Engineering

# %% [cell 6 - code]
WR_B1, WR_B2, WR_B3, WR_B4, WR_B5 = 8.5, 10.75, 11.0, 11.25, 12.0
WF_PERIOD, LG_PERIOD = 0.55, 4.25

def build_features(df_in):
    LG = df_in["LG"].to_numpy(float)
    WR = df_in["WR"].to_numpy(float)
    LR = df_in["LR"].to_numpy(float)
    WF = df_in["WF"].to_numpy(float)
    wr_zone = np.where(WR < WR_B1, 0, np.where(WR < WR_B2, 1, np.where(WR < WR_B3, 2, np.where(WR < WR_B4, 3, np.where(WR < WR_B5, 4, 5))))).astype(float)
    f = {
        "LG": LG, "WR": WR, "LR": LR, "WF": WF,
        "LG2": LG**2, "WR2": WR**2, "LR2": LR**2, "WF2": WF**2,
        "LG3": LG**3, "WR3": WR**3, "WF3": WF**3,
        "LG_WR": LG*WR, "LG_LR": LG*LR, "LG_WF": LG*WF, "WR_LR": WR*LR, "WR_WF": WR*WF, "LR_WF": LR*WF,
        "LG_WR_WF": LG*WR*WF, "LG_WR_LR_WF": LG*WR*LR*WF,
        "LG2_WR": LG**2*WR, "WR2_LG": WR**2*LG,
        "WR_b1": (WR>WR_B1).astype(float), "WR_b2": (WR>WR_B2).astype(float), "WR_b3": (WR>WR_B3).astype(float),
        "WR_b4": (WR>WR_B4).astype(float), "WR_b5": (WR>WR_B5).astype(float),
        "WR_d11": WR-11.0, "WR_d1125": np.clip(WR-11.25, 0, None),
        "WR_zone": wr_zone, "zone_LG": wr_zone*LG, "zone_LR": wr_zone*LR,
        "LR_flag": (LR>3.1).astype(float), "LR_delta": LR-3.0, "LR_zone": (LR>3.1).astype(float)*wr_zone,
        "LR_WR_d11": (LR-3.0)*(WR-11.0), "LR_WF2": LR*WF**2,
        "logWR": np.log1p(WR), "logLG": np.log(LG), "logWF": np.log(WF),
        "WR_WFr": WR/(WF+0.1), "LG_WRr": LG/(WR+0.1), "WR_LGr": WR/(LG+0.1),
        "WR_b3_LG_WF": (WR>WR_B3).astype(float)*LG*WF,
        "WR_d11_LG": (WR-11.0)*LG, "WR_d11_WF": (WR-11.0)*WF,
        "WF2_LG": WF**2*LG, "LG_WF_zone": LG*WF*wr_zone, "LR_LG_WF_zone": LR*LG*WF*wr_zone,
    }
    for k in [1,2,3]:
        a = k*np.pi*WF/WF_PERIOD
        f[f"sinWF{k}"] = np.sin(a); f[f"cosWF{k}"] = np.cos(a)
    for k in [1,2]:
        a = k*np.pi*LG/LG_PERIOD
        f[f"sinLG{k}"] = np.sin(a); f[f"cosLG{k}"] = np.cos(a)
    for b in [8.5,10.75,11.0,11.25,12.0,12.5]:
        f[f"dWR_{b}"] = WR-b
    return pd.DataFrame(f)

X = build_features(df)
print("Feature matrix:", X.shape)

# %% [cell 7 - markdown]
# ## 4. Train/Test Split

# %% [cell 8 - code]
idx_train, idx_val = train_test_split(np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE, stratify=df.is_resonant)
df_train = df.iloc[idx_train].reset_index(drop=True)
df_val = df.iloc[idx_val].reset_index(drop=True)
X_train = build_features(df_train)
X_val = build_features(df_val)
y_train = df_train.is_resonant.to_numpy()
y_val = df_val.is_resonant.to_numpy()
print("Train:", len(df_train), "res", y_train.sum(), "dead", (y_train==0).sum())
print("Val:", len(df_val), "res", y_val.sum(), "dead", (y_val==0).sum())

# %% [cell 9 - markdown]
# ## 5. Resonant-Favoring Classifier
# 
# This threshold is tuned to keep resonant recall high, because the use case is searching for maximum bandwidth. Missing a resonant high-BW design is more costly than suggesting a few false positives.

# %% [cell 10 - code]
def tune_resonant_threshold(y_true, prob, min_res_recall=MIN_RESONANT_RECALL):
    rows = []
    for t in np.arange(0.10, 0.91, 0.01):
        pred = (prob >= t).astype(int)
        rec_res = recall_score(y_true, pred, pos_label=1, zero_division=0)
        rec_dead = recall_score(y_true, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        penalty = max(0.0, min_res_recall - rec_res)
        objective = 0.45*f1 + 0.25*bacc + 0.20*rec_res + 0.10*acc - 0.50*penalty
        rows.append({"threshold": t, "accuracy": acc, "balanced_accuracy": bacc, "f1": f1, "resonant_recall": rec_res, "dead_recall": rec_dead, "objective": objective})
    table = pd.DataFrame(rows)
    feasible = table[table.resonant_recall >= min_res_recall]
    best = (feasible if len(feasible) else table).sort_values(["objective", "f1", "balanced_accuracy"], ascending=False).iloc[0]
    return float(best.threshold), table

def fit_classifier(X_train, y_train):
    base = [
        ("ET", ExtraTreesClassifier(n_estimators=700, max_features="sqrt", min_samples_leaf=1, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
        ("RF", RandomForestClassifier(n_estimators=500, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1)),
        ("HGB", HistGradientBoostingClassifier(max_iter=450, learning_rate=0.045, max_leaf_nodes=31, l2_regularization=0.02, class_weight="balanced", random_state=RANDOM_STATE)),
    ]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.DataFrame(index=np.arange(len(X_train)))
    for fold, (tr, va) in enumerate(skf.split(X_train, y_train), 1):
        for name, model in base:
            m = deepcopy(model)
            m.fit(X_train.iloc[tr], y_train[tr])
            oof.loc[va, name] = m.predict_proba(X_train.iloc[va])[:,1]
        print("fold", fold, "done")
    meta = LogisticRegressionCV(Cs=20, cv=5, class_weight="balanced", scoring="f1", max_iter=3000, random_state=RANDOM_STATE)
    meta.fit(oof, y_train)
    oof_prob = meta.predict_proba(oof)[:,1]
    threshold, threshold_table = tune_resonant_threshold(y_train, oof_prob)
    final_base = []
    for name, model in base:
        m = deepcopy(model)
        m.fit(X_train, y_train)
        final_base.append((name, m))
    return {"base": final_base, "meta": meta, "threshold": threshold, "oof_prob": oof_prob, "threshold_table": threshold_table}

clf = fit_classifier(X_train, y_train)
print("Chosen resonant-favoring threshold:", clf["threshold"])
clf["threshold_table"].to_csv(OUT_DIR / "resonant_threshold_search.csv", index=False)
display(clf["threshold_table"].sort_values("objective", ascending=False).head(10))

# %% [cell 11 - markdown]
# ## 6. Classifier Validation

# %% [cell 12 - code]
def predict_res_prob(clf, X_eval):
    base_prob = pd.DataFrame({name: model.predict_proba(X_eval)[:,1] for name, model in clf["base"]})
    return clf["meta"].predict_proba(base_prob)[:,1]

def report_classifier(y_true, prob, threshold, label):
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0,1])
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, prob),
        "dead_recall": recall_score(y_true, pred, pos_label=0, zero_division=0),
        "resonant_recall": recall_score(y_true, pred, pos_label=1, zero_division=0),
        "threshold": threshold,
    }
    print(label, json.dumps(metrics, indent=2))
    print(cm)
    print(classification_report(y_true, pred, labels=[0,1], target_names=["Dead", "Resonant"], zero_division=0))
    return metrics, cm

val_prob = predict_res_prob(clf, X_val)
val_class_metrics, val_cm = report_classifier(y_val, val_prob, clf["threshold"], "Validation")

# %% [cell 13 - markdown]
# ## 7. Train BW / FL / FU Regressors on Resonant Target-Window Data

# %% [cell 14 - code]
def fit_regressors(df_train):
    # Prefer real operating band data. If a future dataset has too few points, fall back to all resonant rows.
    train_res = df_train[(df_train.BW > 0) & (df_train.FL >= TARGET_FL_MIN) & (df_train.FU <= TARGET_FU_MAX)].reset_index(drop=True)
    if len(train_res) < 100:
        train_res = df_train[df_train.BW > 0].reset_index(drop=True)
    Xr = build_features(train_res)
    regs = {}
    for target in ["BW", "FL", "FU"]:
        models = [
            ("ET", ExtraTreesRegressor(n_estimators=800, max_features="sqrt", min_samples_leaf=1, random_state=RANDOM_STATE, n_jobs=-1)),
            ("RF", RandomForestRegressor(n_estimators=600, max_features="sqrt", min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
            ("HGB", HistGradientBoostingRegressor(max_iter=500, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=0.02, random_state=RANDOM_STATE)),
        ]
        fitted = []
        for name, model in models:
            model.fit(Xr, train_res[target].to_numpy())
            fitted.append((name, model))
        regs[target] = fitted
    return regs, train_res

def predict_target(regs, X_eval, target):
    weights = np.array([0.40, 0.35, 0.25])
    preds = np.column_stack([m.predict(X_eval) for _, m in regs[target]])
    return preds @ weights

regs, reg_train_data = fit_regressors(df_train)
print("Regression training rows:", len(reg_train_data))

val_res_mask = (df_val.BW > 0) & (df_val.FL >= TARGET_FL_MIN) & (df_val.FU <= TARGET_FU_MAX)
if val_res_mask.sum() < 10:
    val_res_mask = df_val.BW > 0
Xv_res = X_val.loc[val_res_mask]
reg_metrics = {}
for target in ["BW", "FL", "FU"]:
    yt = df_val.loc[val_res_mask, target].to_numpy()
    yp = predict_target(regs, Xv_res, target)
    reg_metrics[target] = {"r2": r2_score(yt, yp), "mae": mean_absolute_error(yt, yp), "rmse": mean_squared_error(yt, yp)**0.5}
print(json.dumps(reg_metrics, indent=2))

# %% [cell 15 - markdown]
# ## 8. Rank All Designs by Expected High Bandwidth

# %% [cell 16 - code]
def make_grid():
    rows=[]
    for lg in LG_VALS:
        for wr in WR_VALS:
            for lr in LR_VALS:
                for wf in WF_VALS:
                    rows.append({"LG": lg, "WR": wr, "LR": lr, "WF": wf})
    return pd.DataFrame(rows)

grid = make_grid()
Xg = build_features(grid)
grid["prob_resonant"] = predict_res_prob(clf, Xg)
grid["pred_BW"] = np.maximum(0, predict_target(regs, Xg, "BW"))
grid["pred_FL"] = predict_target(regs, Xg, "FL")
grid["pred_FU"] = predict_target(regs, Xg, "FU")
grid["pred_in_4p5_9GHz"] = ((grid.pred_FL >= TARGET_FL_MIN) & (grid.pred_FU <= TARGET_FU_MAX)).astype(int)
grid["expected_BW"] = grid.prob_resonant * grid.pred_BW
# practical score favors high BW, high resonant probability, and frequency window fit
grid["selection_score"] = grid.expected_BW * (0.75 + 0.25*grid.pred_in_4p5_9GHz)

known = set(zip(df.LG.round(3), df.WR.round(3), df.LR.round(3), df.WF.round(3)))
grid["known_in_data"] = [tuple(x) in known for x in zip(grid.LG.round(3), grid.WR.round(3), grid.LR.round(3), grid.WF.round(3))]

top_all = grid.sort_values("selection_score", ascending=False).head(50)
top_untested = grid[~grid.known_in_data].sort_values("selection_score", ascending=False).head(50)

top_all.to_csv(OUT_DIR / "top_50_high_bw_all_grid.csv", index=False)
top_untested.to_csv(OUT_DIR / "top_50_high_bw_untested.csv", index=False)

print("Top all-grid designs")
display(top_all.head(20))
print("Top untested designs")
display(top_untested.head(20))

# %% [cell 17 - markdown]
# ## 9. Save Model and Summary

# %% [cell 18 - code]
summary = {
    "goal": "favor resonant high-bandwidth candidates in 4.5-9 GHz window",
    "rows": int(len(df)),
    "resonant": int(df.is_resonant.sum()),
    "dead": int((df.is_resonant == 0).sum()),
    "target_window_rows": int(df.in_target_window.sum()),
    "threshold": float(clf["threshold"]),
    "validation_classifier": {k: float(v) for k,v in val_class_metrics.items()},
    "validation_regression": {t: {k: float(v) for k,v in vals.items()} for t, vals in reg_metrics.items()},
    "best_known_data_row": df.sort_values("BW", ascending=False).iloc[0][["LG", "WR", "LR", "WF", "FL", "FU", "BW"]].to_dict(),
    "best_predicted_grid_row": top_all.iloc[0][["LG", "WR", "LR", "WF", "prob_resonant", "pred_FL", "pred_FU", "pred_BW", "expected_BW", "selection_score", "known_in_data"]].to_dict(),
    "best_predicted_untested_row": top_untested.iloc[0][["LG", "WR", "LR", "WF", "prob_resonant", "pred_FL", "pred_FU", "pred_BW", "expected_BW", "selection_score"]].to_dict(),
}

(OUT_DIR / "high_bw_model_summary.json").write_text(json.dumps(summary, indent=2))
pd.DataFrame([summary["validation_classifier"]]).to_csv(OUT_DIR / "high_bw_classifier_metrics.csv", index=False)

with open(OUT_DIR / "high_bw_resonant_model.pkl", "wb") as f:
    pickle.dump({"classifier": clf, "regressors": regs, "summary": summary}, f)

print(json.dumps(summary, indent=2))
