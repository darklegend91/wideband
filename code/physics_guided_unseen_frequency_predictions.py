# Exported from active_val_physics_guided_outputs/physics_guided_unseen_frequency_predictions.ipynb
# Generated for the physics_guided_files bundle.


# %% [cell 0 - markdown]
# # Physics-Guided Unseen Frequency Predictions
# 
# This notebook creates candidate antenna design points using simple microstrip/wavelength physics heuristics, checks whether each point is already present in the old dataset, removes known points, and then predicts `FL`, `FU`, and `BW` using the trained high-bandwidth resonant model.
# 
# The physics formulas here are approximate guide rules, not a replacement for EM simulation. They are used to choose smarter candidate points before ML prediction.

# %% [cell 1 - code]
from pathlib import Path
import json
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

PROJECT_DIR = Path.cwd()
DATA_PATH = PROJECT_DIR / "UBW-data-final.xlsx"
MODEL_PATH = PROJECT_DIR / "active_val_high_bw_resonant_outputs" / "high_bw_resonant_model.pkl"
OUT_DIR = PROJECT_DIR / "active_val_physics_guided_outputs"
OUT_DIR.mkdir(exist_ok=True)

# Fixed antenna constants from the project description/readme.
C_MM_PER_NS = 300.0
PATCH_LENGTH_MM = 11.0
PATCH_WIDTH_MM = 20.0
SUBSTRATE_HEIGHT_MM = 1.6
TARGET_FL_MIN = 4.5
TARGET_FU_MAX = 9.0
TARGET_FC_MIN = 5.0
TARGET_FC_MAX = 8.5

LG_VALS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
WR_VALS = [0.75, 2.25, 2.5, 7.5, 8.5, 9.25, 9.5, 10.0, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75, 12.0, 12.25, 12.5, 12.75]
LR_VALS = [3.0, 3.3]
WF_VALS = [round(x, 1) for x in np.arange(2.0, 5.0 + 0.001, 0.1)]

print("Dataset:", DATA_PATH)
print("Model:", MODEL_PATH)
print("Output:", OUT_DIR)

# %% [cell 2 - markdown]
# ## 1. Load Old Dataset and Trained Model

# %% [cell 3 - code]
df = pd.read_excel(DATA_PATH, header=0, usecols=[0,1,2,3,4,5,6,7])
df.columns = ["SrNo", "LG", "WR", "LR", "WF", "FL", "FU", "BW"]
df = df.dropna(subset=["LG", "WR", "LR", "WF", "FL", "FU", "BW"]).copy()
df[["LG", "WR", "LR", "WF", "FL", "FU", "BW"]] = df[["LG", "WR", "LR", "WF", "FL", "FU", "BW"]].astype(float)
df = (df.sort_values("BW", ascending=False)
        .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
        .reset_index(drop=True))
df["is_resonant"] = (df.BW > 0).astype(int)
df["fc"] = np.where(df.BW > 0, (df.FL + df.FU) / 2.0, np.nan)
df["frac_bw"] = np.where(df.BW > 0, df.BW / df.fc, np.nan)

with open(MODEL_PATH, "rb") as f:
    artifact = pickle.load(f)
clf = artifact["classifier"]
regs = artifact["regressors"]

known_keys = set(zip(df.LG.round(3), df.WR.round(3), df.LR.round(3), df.WF.round(3)))
print("Old unique dataset rows:", len(df))
print("Resonant rows:", int(df.is_resonant.sum()))
print("Dead rows:", int((df.is_resonant == 0).sum()))
print("Known keys:", len(known_keys))
display(df.sort_values("BW", ascending=False).head(10)[["LG","WR","LR","WF","FL","FU","BW","fc","frac_bw"]])

# %% [cell 4 - markdown]
# ## 2. Estimate Physics Quantities From the Existing Resonant Data
# 
# For a rectangular microstrip mode, a first-order frequency relation is:
# 
# `f ≈ 150 / (L_eff_mm * sqrt(eps_eff))` GHz
# 
# Using the known patch length and observed resonant center frequencies, we estimate an empirical effective permittivity. This is only a calibration heuristic.

# %% [cell 5 - code]
res = df[df.BW > 0].copy()
# Rearranged from f_GHz = 150 / (L_mm * sqrt(eps_eff)).
res["eps_eff_est"] = (150.0 / (PATCH_LENGTH_MM * res.fc)) ** 2
EPS_EFF_MEDIAN = float(res.eps_eff_est.replace([np.inf, -np.inf], np.nan).dropna().median())
EPS_EFF_CLIPPED = float(np.clip(EPS_EFF_MEDIAN, 1.2, 12.0))

# Guided wavelength in mm: lambda_g = 300 / (f_GHz * sqrt(eps_eff)).
res["lambda_g_mm"] = C_MM_PER_NS / (res.fc * np.sqrt(EPS_EFF_CLIPPED))
res["WR_lambda_ratio"] = res.WR / res.lambda_g_mm
res["LG_lambda_ratio"] = res.LG / res.lambda_g_mm
res["WF_lambda_ratio"] = res.WF / res.lambda_g_mm

high = res.sort_values("BW", ascending=False).head(max(30, int(0.05 * len(res))))
physics_targets = {
    "eps_eff_median_raw": EPS_EFF_MEDIAN,
    "eps_eff_used": EPS_EFF_CLIPPED,
    "high_bw_fc_median": float(high.fc.median()),
    "high_bw_frac_bw_median": float(high.frac_bw.median()),
    "high_bw_WR_lambda_ratio_median": float(high.WR_lambda_ratio.median()),
    "high_bw_LG_lambda_ratio_median": float(high.LG_lambda_ratio.median()),
    "high_bw_WF_lambda_ratio_median": float(high.WF_lambda_ratio.median()),
    "best_known": high.iloc[0][["LG","WR","LR","WF","FL","FU","BW","fc","frac_bw"]].to_dict(),
}
print(json.dumps(physics_targets, indent=2))
(OUT_DIR / "physics_calibration_summary.json").write_text(json.dumps(physics_targets, indent=2))

# %% [cell 6 - markdown]
# ## 3. Feature Engineering for the Trained Model

# %% [cell 7 - code]
WR_B1, WR_B2, WR_B3, WR_B4, WR_B5 = 8.5, 10.75, 11.0, 11.25, 12.0
WF_PERIOD, LG_PERIOD = 0.55, 4.25

def build_features(df_in):
    LG = df_in["LG"].to_numpy(float); WR = df_in["WR"].to_numpy(float)
    LR = df_in["LR"].to_numpy(float); WF = df_in["WF"].to_numpy(float)
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

# %% [cell 8 - markdown]
# ## 4. Generate Physics-Guided Candidate Points
# 
# We compute guided-wavelength ratios for the full grid and keep candidates close to the ratios observed in the best known high-BW designs. Then we remove every point already present in the old dataset.

# %% [cell 9 - code]
rows=[]
for lg in LG_VALS:
    for wr in WR_VALS:
        for lr in LR_VALS:
            for wf in WF_VALS:
                rows.append({"LG":lg,"WR":wr,"LR":lr,"WF":wf})
grid = pd.DataFrame(rows)
grid["known_in_old_dataset"] = [tuple(x) in known_keys for x in zip(grid.LG.round(3), grid.WR.round(3), grid.LR.round(3), grid.WF.round(3))]

# Physics estimates across target center frequencies.
fc_grid = np.linspace(TARGET_FC_MIN, TARGET_FC_MAX, 36)
physics_rows=[]
for _, row in grid.iterrows():
    for fc in fc_grid:
        lambda_g = C_MM_PER_NS / (fc * np.sqrt(EPS_EFF_CLIPPED))
        wr_ratio = row.WR / lambda_g
        lg_ratio = row.LG / lambda_g
        wf_ratio = row.WF / lambda_g
        # Score closeness to high-bandwidth empirical wavelength ratios.
        score = np.exp(-((wr_ratio - physics_targets["high_bw_WR_lambda_ratio_median"]) / 0.08) ** 2)
        score *= np.exp(-((lg_ratio - physics_targets["high_bw_LG_lambda_ratio_median"]) / 0.07) ** 2)
        score *= np.exp(-((wf_ratio - physics_targets["high_bw_WF_lambda_ratio_median"]) / 0.05) ** 2)
        physics_rows.append({
            "LG": row.LG, "WR": row.WR, "LR": row.LR, "WF": row.WF,
            "physics_fc": fc, "lambda_g_mm": lambda_g,
            "WR_lambda_ratio": wr_ratio, "LG_lambda_ratio": lg_ratio, "WF_lambda_ratio": wf_ratio,
            "physics_score": score,
            "known_in_old_dataset": row.known_in_old_dataset,
        })
physics_df = pd.DataFrame(physics_rows)
# Keep best physics center-frequency match per design.
physics_best = (physics_df.sort_values("physics_score", ascending=False)
                         .drop_duplicates(subset=["LG","WR","LR","WF"], keep="first")
                         .reset_index(drop=True))
physics_unseen = physics_best[~physics_best.known_in_old_dataset].copy()
physics_unseen = physics_unseen.sort_values("physics_score", ascending=False).reset_index(drop=True)
print("Full grid:", len(grid))
print("Unseen physics candidates:", len(physics_unseen))
display(physics_unseen.head(20))

# %% [cell 10 - markdown]
# ## 5. Predict Frequencies and Bandwidth on Physics-Guided Unseen Points

# %% [cell 11 - code]
def predict_res_prob(X):
    base_prob = pd.DataFrame({name: model.predict_proba(X)[:,1] for name, model in clf["base"]})
    return clf["meta"].predict_proba(base_prob)[:,1]

def predict_reg(X, target):
    preds = np.column_stack([model.predict(X) for _, model in regs[target]])
    weights = np.array([0.40, 0.35, 0.25])
    return preds @ weights, preds.std(axis=1)

Xc = build_features(physics_unseen)
physics_unseen["prob_resonant"] = predict_res_prob(Xc)
physics_unseen["pred_BW"], physics_unseen["bw_model_std"] = predict_reg(Xc, "BW")
physics_unseen["pred_FL"], physics_unseen["fl_model_std"] = predict_reg(Xc, "FL")
physics_unseen["pred_FU"], physics_unseen["fu_model_std"] = predict_reg(Xc, "FU")
physics_unseen["pred_BW"] = physics_unseen.pred_BW.clip(lower=0)
physics_unseen["pred_fc"] = (physics_unseen.pred_FL + physics_unseen.pred_FU) / 2.0
physics_unseen["pred_frac_bw"] = physics_unseen.pred_BW / physics_unseen.pred_fc
physics_unseen["pred_in_4p5_9GHz"] = ((physics_unseen.pred_FL >= TARGET_FL_MIN) & (physics_unseen.pred_FU <= TARGET_FU_MAX)).astype(int)
physics_unseen["expected_BW"] = physics_unseen.prob_resonant * physics_unseen.pred_BW
physics_unseen["confidence_penalty"] = 1.0 / (1.0 + physics_unseen.bw_model_std)
physics_unseen["combined_score"] = physics_unseen.expected_BW * physics_unseen.confidence_penalty * (0.70 + 0.30*physics_unseen.physics_score) * (0.80 + 0.20*physics_unseen.pred_in_4p5_9GHz)

ranked = physics_unseen.sort_values("combined_score", ascending=False).reset_index(drop=True)
ranked.insert(0, "rank", np.arange(1, len(ranked)+1))
ranked.head(100).to_csv(OUT_DIR / "physics_guided_top_100_unseen_predictions.csv", index=False)
ranked.head(25).to_csv(OUT_DIR / "physics_guided_top_25_unseen_predictions.csv", index=False)

top25 = ranked.head(25).copy()
top25["old_dataset_match"] = [tuple(x) in known_keys for x in zip(top25.LG.round(3), top25.WR.round(3), top25.LR.round(3), top25.WF.round(3))]
assert not top25.old_dataset_match.any()

display(top25[["rank","LG","WR","LR","WF","physics_fc","physics_score","prob_resonant","pred_FL","pred_FU","pred_BW","expected_BW","combined_score","old_dataset_match"]])

# %% [cell 12 - markdown]
# ## 6. Save Final Summary

# %% [cell 13 - code]
best = ranked.iloc[0]
summary = {
    "method": "physics-guided grid candidates + old-dataset exclusion + trained ML frequency prediction",
    "all_recommendations_are_unseen": True,
    "top_recommendation_count": 25,
    "old_dataset_rows": int(len(df)),
    "candidate_grid_rows": int(len(grid)),
    "unseen_candidate_rows": int(len(physics_unseen)),
    "eps_eff_used": EPS_EFF_CLIPPED,
    "best_recommendation": best[["LG","WR","LR","WF","physics_fc","physics_score","prob_resonant","pred_FL","pred_FU","pred_BW","expected_BW","combined_score"]].to_dict(),
    "best_known_dataset_row": df.sort_values("BW", ascending=False).iloc[0][["LG","WR","LR","WF","FL","FU","BW"]].to_dict(),
}
(OUT_DIR / "physics_guided_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
