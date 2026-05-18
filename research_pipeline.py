"""
Single-file research workflow for wideband antenna bandwidth prediction.

This script trains the consolidated model, evaluates base techniques against
the stacked ensemble, and writes publication-ready plots for confusion
matrices, learning curves, accuracy/loss, regression quality, feature
importance, and physics-guided candidate ranking.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import warnings
from copy import deepcopy
from itertools import product
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/wideband-mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/wideband-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer

warnings.filterwarnings("ignore")


RANDOM_STATE = 42
TARGET_FL_MIN = 4.5
TARGET_FU_MAX = 9.0
MIN_RESONANT_RECALL = 0.93

LG_VALS = [4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
WR_VALS = [
    0.75,
    2.25,
    2.5,
    7.5,
    8.5,
    9.25,
    9.5,
    10.0,
    10.5,
    10.75,
    11.0,
    11.25,
    11.5,
    11.75,
    12.0,
    12.25,
    12.5,
    12.75,
]
LR_VALS = [3.0, 3.3]
WF_VALS = [round(x, 1) for x in np.arange(2.0, 5.0 + 0.001, 0.1)]

WR_B1, WR_B2, WR_B3, WR_B4, WR_B5 = 8.5, 10.75, 11.0, 11.25, 12.0
WF_PERIOD, LG_PERIOD = 0.55, 4.25

PLOT_DPI = 180


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "plots": output_dir / "plots",
        "tables": output_dir / "tables",
        "models": output_dir / "models",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def resolve_data_path(path_arg: str | None) -> Path:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    candidates.extend([Path("data/UBW-data-final.xlsx"), Path("UBW-data-final.xlsx")])
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Could not find UBW-data-final.xlsx. Use --data to provide its path.")


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, header=0, usecols=[0, 1, 2, 3, 4, 5, 6, 7])
    df.columns = ["SrNo", "LG", "WR", "LR", "WF", "FL", "FU", "BW"]
    numeric = ["LG", "WR", "LR", "WF", "FL", "FU", "BW"]
    df = df.dropna(subset=numeric).copy()
    df[numeric] = df[numeric].astype(float)
    df = (
        df.sort_values("BW", ascending=False)
        .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
        .reset_index(drop=True)
    )
    df["is_resonant"] = (df["BW"] > 0).astype(int)
    df["in_target_window"] = (
        (df["BW"] > 0) & (df["FL"] >= TARGET_FL_MIN) & (df["FU"] <= TARGET_FU_MAX)
    ).astype(int)
    df["fc"] = np.where(df["BW"] > 0, (df["FL"] + df["FU"]) / 2.0, np.nan)
    df["frac_bw"] = np.where(df["BW"] > 0, df["BW"] / df["fc"], np.nan)
    df["source"] = "original"
    return df


def load_feedback_data(path: Path) -> pd.DataFrame:
    """
    Load simulator feedback rows. The file can contain either clean training
    columns LG, WR, LR, WF, FL, FU, BW or exported prediction columns plus
    Actual FL, FU, BW.
    """
    raw = pd.read_csv(path)
    if len(raw.columns) == 1 and raw.columns[0].strip().lower() not in {"lg", "wr", "lr", "wf"}:
        # Spreadsheet exports sometimes include a title line before the real
        # CSV header, for example "feedback_simulations".
        raw = pd.read_csv(path, skiprows=1)
    raw.columns = [str(col).strip() for col in raw.columns]
    rename_map = {
        "Pred Freq(Lower) GHz": "pred_FL",
        "Pred Freq(Upper) GHz": "pred_FU",
        "Pred BW GHz": "pred_BW",
        "Probability resonant": "pred_prob_resonant",
        "Expected BW GHz": "pred_expected_BW",
        "Combined score": "pred_combined_score",
        "Known in old dataset": "known_in_old_dataset",
        "Actual FL": "FL",
        "Actual FU": "FU",
        "Actual BW": "BW",
    }
    df = raw.rename(columns=rename_map).copy()
    required = ["LG", "WR", "LR", "WF", "FL", "FU", "BW"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Feedback file {path} is missing required columns: {missing}")
    df = df.dropna(subset=required).copy()
    df[required] = df[required].astype(float)
    df["SrNo"] = np.arange(1, len(df) + 1)
    df["source"] = "feedback"
    keep_cols = ["SrNo", "LG", "WR", "LR", "WF", "FL", "FU", "BW", "source"]
    optional_cols = [
        "physics_fc",
        "pred_FL",
        "pred_FU",
        "pred_BW",
        "pred_prob_resonant",
        "pred_expected_BW",
        "pred_combined_score",
        "known_in_old_dataset",
    ]
    return df[keep_cols + [col for col in optional_cols if col in df.columns]]


def finalize_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source_priority"] = df["source"].map({"original": 0, "feedback": 1}).fillna(0)
    df = (
        df.sort_values(["source_priority", "BW"], ascending=[False, False])
        .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
        .drop(columns=["source_priority"])
        .reset_index(drop=True)
    )
    df["is_resonant"] = (df["BW"] > 0).astype(int)
    df["in_target_window"] = (
        (df["BW"] > 0) & (df["FL"] >= TARGET_FL_MIN) & (df["FU"] <= TARGET_FU_MAX)
    ).astype(int)
    df["fc"] = np.where(df["BW"] > 0, (df["FL"] + df["FU"]) / 2.0, np.nan)
    df["frac_bw"] = np.where(df["BW"] > 0, df["BW"] / df["fc"], np.nan)
    return df


def combine_with_feedback(df: pd.DataFrame, feedback_path_arg: str | None) -> tuple[pd.DataFrame, int, Path | None]:
    candidates = []
    if feedback_path_arg:
        candidates.append(Path(feedback_path_arg))
    else:
        candidates.append(Path("data/feedback_simulations.csv"))

    for path in candidates:
        if path.exists():
            feedback = load_feedback_data(path)
            merged = finalize_training_frame(pd.concat([df, feedback], ignore_index=True, sort=False))
            return merged, len(feedback), path

    return finalize_training_frame(df), 0, None


def build_features(df_in: pd.DataFrame) -> pd.DataFrame:
    LG = df_in["LG"].to_numpy(float)
    WR = df_in["WR"].to_numpy(float)
    LR = df_in["LR"].to_numpy(float)
    WF = df_in["WF"].to_numpy(float)
    wr_zone = np.where(
        WR < WR_B1,
        0,
        np.where(
            WR < WR_B2,
            1,
            np.where(WR < WR_B3, 2, np.where(WR < WR_B4, 3, np.where(WR < WR_B5, 4, 5))),
        ),
    ).astype(float)

    f = {
        "LG": LG,
        "WR": WR,
        "LR": LR,
        "WF": WF,
        "LG2": LG**2,
        "WR2": WR**2,
        "LR2": LR**2,
        "WF2": WF**2,
        "LG3": LG**3,
        "WR3": WR**3,
        "WF3": WF**3,
        "LG_WR": LG * WR,
        "LG_LR": LG * LR,
        "LG_WF": LG * WF,
        "WR_LR": WR * LR,
        "WR_WF": WR * WF,
        "LR_WF": LR * WF,
        "LG_WR_WF": LG * WR * WF,
        "LG_WR_LR_WF": LG * WR * LR * WF,
        "LG2_WR": LG**2 * WR,
        "WR2_LG": WR**2 * LG,
        "WR_b1": (WR > WR_B1).astype(float),
        "WR_b2": (WR > WR_B2).astype(float),
        "WR_b3": (WR > WR_B3).astype(float),
        "WR_b4": (WR > WR_B4).astype(float),
        "WR_b5": (WR > WR_B5).astype(float),
        "WR_d11": WR - 11.0,
        "WR_d1125": np.clip(WR - 11.25, 0, None),
        "WR_zone": wr_zone,
        "zone_LG": wr_zone * LG,
        "zone_LR": wr_zone * LR,
        "LR_flag": (LR > 3.1).astype(float),
        "LR_delta": LR - 3.0,
        "LR_zone": (LR > 3.1).astype(float) * wr_zone,
        "LR_WR_d11": (LR - 3.0) * (WR - 11.0),
        "LR_WF2": LR * WF**2,
        "logWR": np.log1p(WR),
        "logLG": np.log(LG),
        "logWF": np.log(WF),
        "logLR": np.log(LR),
        "WR_WFr": WR / (WF + 0.1),
        "LG_WRr": LG / (WR + 0.1),
        "WR_LGr": WR / (LG + 0.1),
        "WR_b3_LG_WF": (WR > WR_B3).astype(float) * LG * WF,
        "WR_d11_LG": (WR - 11.0) * LG,
        "WR_d11_WF": (WR - 11.0) * WF,
        "WF2_LG": WF**2 * LG,
        "LG_WF_zone": LG * WF * wr_zone,
        "LR_LG_WF_zone": LR * LG * WF * wr_zone,
        "WR_b3_LR_WF": (WR > WR_B3).astype(float) * LR * WF,
        "WR_d11_LR": (WR - 11.0) * LR,
        "LR_LG_WF": LR * LG * WF,
    }
    for k in [1, 2, 3]:
        a = k * np.pi * WF / WF_PERIOD
        f[f"sinWF{k}"] = np.sin(a)
        f[f"cosWF{k}"] = np.cos(a)
    for k in [1, 2]:
        a = k * np.pi * LG / LG_PERIOD
        f[f"sinLG{k}"] = np.sin(a)
        f[f"cosLG{k}"] = np.cos(a)
    f["sinWF1_LG"] = np.sin(np.pi * WF / WF_PERIOD) * LG
    f["cosWF1_LG"] = np.cos(np.pi * WF / WF_PERIOD) * LG
    f["sinWF1_LR"] = np.sin(np.pi * WF / WF_PERIOD) * LR
    f["cosWF1_LR"] = np.cos(np.pi * WF / WF_PERIOD) * LR
    for b in [8.5, 10.75, 11.0, 11.25, 12.0, 12.5]:
        f[f"dWR_{b}"] = WR - b
    return pd.DataFrame(f)


def lookup_key_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    return df_in[["LG", "WR", "LR", "WF"]].round(3)


def build_vlookup_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["LG", "WR", "LR", "WF", "FL", "FU", "BW", "is_resonant", "in_target_window", "source"]
    lookup = df[cols].copy()
    lookup[["LG", "WR", "LR", "WF"]] = lookup_key_frame(lookup)
    return (
        lookup.sort_values(["source", "BW"], ascending=[True, False])
        .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
        .reset_index(drop=True)
    )


def attach_vlookup_predictions(candidates: pd.DataFrame, lookup_table: pd.DataFrame) -> pd.DataFrame:
    left = candidates.copy()
    left[["LG", "WR", "LR", "WF"]] = lookup_key_frame(left)
    lookup = lookup_table.rename(
        columns={
            "FL": "lookup_FL",
            "FU": "lookup_FU",
            "BW": "lookup_BW",
            "is_resonant": "lookup_is_resonant",
            "in_target_window": "lookup_in_target_window",
            "source": "lookup_source",
        }
    )
    merged = left.merge(
        lookup,
        on=["LG", "WR", "LR", "WF"],
        how="left",
    )
    merged["known_in_lookup"] = merged["lookup_BW"].notna()
    return merged


def apply_vlookup_to_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    out = predictions.copy()
    known = out["known_in_lookup"].fillna(False).to_numpy(bool)
    out["prob_resonant_lookup_assisted"] = np.where(known, out["lookup_is_resonant"], out["prob_resonant"])
    out["pred_FL_lookup_assisted"] = np.where(known, out["lookup_FL"], out["pred_FL"])
    out["pred_FU_lookup_assisted"] = np.where(known, out["lookup_FU"], out["pred_FU"])
    out["pred_BW_lookup_assisted"] = np.where(known, out["lookup_BW"], out["pred_BW"])
    return out


def evaluate_vlookup_known_data(lookup_table: pd.DataFrame) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    y_true = lookup_table["is_resonant"].to_numpy()
    prob = lookup_table["is_resonant"].to_numpy(float)
    class_metrics = classifier_metrics(y_true, prob, 0.5)
    reg_metrics = {}
    res = lookup_table[lookup_table["BW"] > 0].copy()
    for target in ["BW", "FL", "FU"]:
        reg_metrics[target] = {
            "r2": float(r2_score(res[target], res[target])),
            "mae": float(mean_absolute_error(res[target], res[target])),
            "rmse": float(mean_squared_error(res[target], res[target]) ** 0.5),
        }
    return class_metrics, reg_metrics


def evaluate_lookup_assisted_validation(
    df_val: pd.DataFrame,
    val_prob: np.ndarray,
    pred_table: pd.DataFrame,
    lookup_table: pd.DataFrame,
    threshold: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]], pd.DataFrame]:
    val_predictions = df_val[["LG", "WR", "LR", "WF", "FL", "FU", "BW", "is_resonant"]].copy()
    val_predictions["prob_resonant"] = val_prob
    pred_cols = pred_table[["LG", "WR", "LR", "WF", "pred_BW", "pred_FL", "pred_FU"]]
    val_predictions = val_predictions.merge(pred_cols, on=["LG", "WR", "LR", "WF"], how="left")
    val_predictions = attach_vlookup_predictions(val_predictions, lookup_table)
    val_predictions = apply_vlookup_to_predictions(val_predictions)

    class_metrics = classifier_metrics(
        val_predictions["is_resonant"].to_numpy(),
        val_predictions["prob_resonant_lookup_assisted"].to_numpy(float),
        threshold,
    )

    reg_metrics = {}
    val_res = val_predictions[(val_predictions["BW"] > 0) & val_predictions["pred_BW_lookup_assisted"].notna()]
    for target in ["BW", "FL", "FU"]:
        reg_metrics[target] = {
            "r2": float(r2_score(val_res[target], val_res[f"pred_{target}_lookup_assisted"])),
            "mae": float(mean_absolute_error(val_res[target], val_res[f"pred_{target}_lookup_assisted"])),
            "rmse": float(mean_squared_error(val_res[target], val_res[f"pred_{target}_lookup_assisted"]) ** 0.5),
        }
    return class_metrics, reg_metrics, val_predictions


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, prob))


def classifier_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": safe_auc(y_true, prob),
        "loss": float(log_loss(y_true, np.clip(prob, 1e-9, 1 - 1e-9), labels=[0, 1])),
    }


def tune_resonant_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.10, 0.91, 0.01):
        pred = (prob >= threshold).astype(int)
        res_recall = recall_score(y_true, pred, pos_label=1, zero_division=0)
        dead_recall = recall_score(y_true, pred, pos_label=0, zero_division=0)
        bacc = balanced_accuracy_score(y_true, pred)
        f1 = f1_score(y_true, pred, zero_division=0)
        acc = accuracy_score(y_true, pred)
        penalty = max(0.0, MIN_RESONANT_RECALL - res_recall)
        objective = 0.45 * f1 + 0.25 * bacc + 0.20 * res_recall + 0.10 * acc - 0.50 * penalty
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(acc),
                "balanced_accuracy": float(bacc),
                "f1": float(f1),
                "resonant_recall": float(res_recall),
                "dead_recall": float(dead_recall),
                "objective": float(objective),
            }
        )
    table = pd.DataFrame(rows)
    feasible = table[table["resonant_recall"] >= MIN_RESONANT_RECALL]
    best = (feasible if len(feasible) else table).sort_values(
        ["objective", "f1", "balanced_accuracy"], ascending=False
    ).iloc[0]
    return float(best["threshold"]), table


def base_classifiers() -> list[tuple[str, object]]:
    return [
        (
            "ET",
            ExtraTreesClassifier(
                n_estimators=700,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "RF",
            RandomForestClassifier(
                n_estimators=500,
                max_features="sqrt",
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
        (
            "HGB",
            HistGradientBoostingClassifier(
                max_iter=450,
                learning_rate=0.045,
                max_leaf_nodes=31,
                l2_regularization=0.02,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
        ),
    ]


def fit_classifier(X_train: pd.DataFrame, y_train: np.ndarray) -> dict[str, object]:
    base = base_classifiers()
    min_class = int(np.bincount(y_train).min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        raise ValueError("Training data must contain at least two samples from each class.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = pd.DataFrame(index=np.arange(len(X_train)))
    for tr_idx, va_idx in skf.split(X_train, y_train):
        for name, model in base:
            m = deepcopy(model)
            m.fit(X_train.iloc[tr_idx], y_train[tr_idx])                        #type:ignore
            oof.loc[va_idx, name] = m.predict_proba(X_train.iloc[va_idx])[:, 1] #type:ignore

    meta = LogisticRegressionCV(
        Cs=20,
        cv=n_splits,
        class_weight="balanced",
        scoring="f1",
        max_iter=3000,
        random_state=RANDOM_STATE,
    )
    meta.fit(oof, y_train)
    oof_prob = meta.predict_proba(oof)[:, 1]
    threshold, threshold_table = tune_resonant_threshold(y_train, oof_prob)

    final_base = []
    for name, model in base:
        fitted = deepcopy(model)
        fitted.fit(X_train, y_train)                                                                    #type:ignore
        final_base.append((name, fitted))

    return {
        "base": final_base,
        "meta": meta,
        "threshold": threshold,
        "oof_prob": oof_prob,
        "oof_base_prob": oof,
        "threshold_table": threshold_table,
    }


def predict_res_prob(clf: dict[str, object], X_eval: pd.DataFrame) -> np.ndarray:
    base_prob = pd.DataFrame({name: model.predict_proba(X_eval)[:, 1] for name, model in clf["base"]}) #type:ignore
    return clf["meta"].predict_proba(base_prob)[:, 1]                                                   #type:ignore


def fit_regressors(df_train: pd.DataFrame) -> tuple[dict[str, list[tuple[str, object]]], pd.DataFrame]:
    train_res = df_train[
        (df_train["BW"] > 0) & (df_train["FL"] >= TARGET_FL_MIN) & (df_train["FU"] <= TARGET_FU_MAX)
    ].reset_index(drop=True)
    if len(train_res) < 100:
        train_res = df_train[df_train["BW"] > 0].reset_index(drop=True)
    Xr = build_features(train_res)
    regs = {}
    for target in ["BW", "FL", "FU"]:
        models = [
            (
                "ET",
                ExtraTreesRegressor(
                    n_estimators=800,
                    max_features="sqrt",
                    min_samples_leaf=1,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "RF",
                RandomForestRegressor(
                    n_estimators=600,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
            (
                "HGB",
                HistGradientBoostingRegressor(
                    max_iter=500,
                    learning_rate=0.04,
                    max_leaf_nodes=31,
                    l2_regularization=0.02,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
        fitted = []
        for name, model in models:
            model.fit(Xr, train_res[target].to_numpy())
            fitted.append((name, model))
        regs[target] = fitted
    return regs, train_res


def predict_reg(regs: dict[str, list[tuple[str, object]]], X_eval: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    weights = np.array([0.40, 0.35, 0.25])
    preds = np.column_stack([model.predict(X_eval) for _, model in regs[target]])                                               #type:ignore
    return preds @ weights, preds.std(axis=1)


def evaluate_regressors(
    regs: dict[str, list[tuple[str, object]]], df_val: pd.DataFrame
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    val_res = df_val[
        (df_val["BW"] > 0) & (df_val["FL"] >= TARGET_FL_MIN) & (df_val["FU"] <= TARGET_FU_MAX)
    ].reset_index(drop=True)
    if len(val_res) < 10:
        val_res = df_val[df_val["BW"] > 0].reset_index(drop=True)
    Xv = build_features(val_res)
    metrics = {}
    pred_table = val_res[["LG", "WR", "LR", "WF", "FL", "FU", "BW"]].copy()
    for target in ["BW", "FL", "FU"]:
        pred, std = predict_reg(regs, Xv, target)
        pred_table[f"pred_{target}"] = pred
        pred_table[f"{target}_model_std"] = std
        metrics[target] = {
            "r2": float(r2_score(val_res[target], pred)),
            "mae": float(mean_absolute_error(val_res[target], pred)),
            "rmse": float(mean_squared_error(val_res[target], pred) ** 0.5),
        }
    return metrics, pred_table


def train_epoch_monitor(
    df_train: pd.DataFrame, df_val: pd.DataFrame, epochs: int, paths: dict[str, Path]
) -> tuple[pd.DataFrame, dict[str, object]]:
    X_train, X_val = build_features(df_train), build_features(df_val)
    y_train, y_val = df_train["is_resonant"].to_numpy(), df_val["is_resonant"].to_numpy()
    monitor = Pipeline(
        [
            ("scaler", QuantileTransformer(output_distribution="normal", random_state=RANDOM_STATE)),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.005,
                    max_iter=1,
                    warm_start=True,
                    early_stopping=False,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    rows = []
    for epoch in range(1, epochs + 1):
        monitor.fit(X_train, y_train)
        train_prob = monitor.predict_proba(X_train)[:, 1]
        val_prob = monitor.predict_proba(X_val)[:, 1]
        train_pred = (train_prob >= 0.5).astype(int)
        val_pred = (val_prob >= 0.5).astype(int)
        rows.append(
            {
                "epoch": epoch,
                "train_accuracy": accuracy_score(y_train, train_pred),
                "validation_accuracy": accuracy_score(y_val, val_pred),
                "train_loss": log_loss(y_train, np.clip(train_prob, 1e-9, 1 - 1e-9), labels=[0, 1]),
                "validation_loss": log_loss(y_val, np.clip(val_prob, 1e-9, 1 - 1e-9), labels=[0, 1]),
            }
        )
    history = pd.DataFrame(rows)
    final_pred = monitor.predict(X_val)
    final_prob = monitor.predict_proba(X_val)[:, 1]
    history.to_csv(paths["tables"] / "epoch_learning_history.csv", index=False)
    return history, {
        "model": monitor,
        "y_true": y_val,
        "y_pred": final_pred,
        "prob": final_prob,
        "metrics": classifier_metrics(y_val, final_prob, 0.5),
    }


def technique_comparison(
    clf: dict[str, object], X_val: pd.DataFrame, y_val: np.ndarray
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    base_probs = {name: model.predict_proba(X_val)[:, 1] for name, model in clf["base"]}                                                                        #type:ignore
    stacked_prob = predict_res_prob(clf, X_val)
    stages = {
        "Extra Trees": base_probs["ET"],
        "Random Forest": base_probs["RF"],
        "Hist Gradient Boosting": base_probs["HGB"],
        "ET + RF Average": np.column_stack([base_probs["ET"], base_probs["RF"]]).mean(axis=1),
        "ET + RF + HGB Average": np.column_stack([base_probs["ET"], base_probs["RF"], base_probs["HGB"]]).mean(axis=1),
        "Consolidated Stacked Model": stacked_prob,
    }
    rows = []
    for name, prob in stages.items():
        threshold = clf["threshold"] if name == "Consolidated Stacked Model" else 0.5
        rows.append({"technique": name, "threshold": threshold, **classifier_metrics(y_val, prob, threshold)})                                                       #type:ignore
    return pd.DataFrame(rows), stages


def style_ax(ax, title: str | None = None) -> None:
    ax.set_facecolor("#f8fafc")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def save_epoch_plots(history: pd.DataFrame, paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    style_ax(axes[0], "Learning Accuracy Across Epochs")
    axes[0].plot(history["epoch"], history["train_accuracy"], marker="o", label="Train", color="#2563eb")
    axes[0].plot(history["epoch"], history["validation_accuracy"], marker="s", label="Validation", color="#16a34a")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()

    style_ax(axes[1], "Learning Loss Across Epochs")
    axes[1].plot(history["epoch"], history["train_loss"], marker="o", label="Train", color="#ca8a04")
    axes[1].plot(history["epoch"], history["validation_loss"], marker="s", label="Validation", color="#dc2626")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Log Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(paths["plots"] / "epoch_accuracy_loss.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_confusion_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    style_ax(ax, title)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Dead", "Resonant"])
    ax.set_yticks([0, 1], labels=["Dead", "Resonant"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_technique_plots(metrics: pd.DataFrame, paths: dict[str, Path]) -> None:
    labels = metrics["technique"].str.replace("Consolidated Stacked Model", "Consolidated\nStacked").str.replace(
        "Hist Gradient Boosting", "HistGB"
    )
    x = np.arange(len(metrics))

    fig, ax = plt.subplots(figsize=(13, 5.2))
    style_ax(ax, "Technique Comparison: Single Models vs Consolidated Model")
    width = 0.2
    for offset, col, name in [(-1.5, "accuracy", "Accuracy"), (-0.5, "balanced_accuracy", "Balanced Acc"), (0.5, "f1", "F1"), (1.5, "auc", "AUC")]:
        ax.bar(x + offset * width, metrics[col], width=width, label=name)
    ax.set_xticks(x, labels=labels, rotation=18, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(paths["plots"] / "technique_comparison.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    style_ax(axes[0], "Accuracy Progression")
    axes[0].plot(labels, metrics["accuracy"], marker="o", linewidth=2.2, color="#2563eb")
    axes[0].set_ylim(0, 1.05)
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Accuracy")
    for i, val in enumerate(metrics["accuracy"]):
        axes[0].annotate(f"{val:.3f}", (i, val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    style_ax(axes[1], "Log Loss Progression")
    axes[1].plot(labels, metrics["loss"], marker="o", linewidth=2.2, color="#ca8a04")
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_ylabel("Log Loss")
    for i, val in enumerate(metrics["loss"]):
        axes[1].annotate(f"{val:.3f}", (i, val), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(paths["plots"] / "technique_accuracy_loss_progression.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_regression_plots(pred_table: pd.DataFrame, paths: dict[str, Path]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, target in zip(axes, ["BW", "FL", "FU"]):
        style_ax(ax, f"{target}: Actual vs Predicted")
        actual = pred_table[target]
        pred = pred_table[f"pred_{target}"]
        ax.scatter(actual, pred, s=26, alpha=0.75, color="#2563eb", edgecolor="white", linewidth=0.4)
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "--", color="#111827", linewidth=1.2)
        ax.set_xlabel(f"Actual {target} (GHz)")
        ax.set_ylabel(f"Predicted {target} (GHz)")
    fig.tight_layout()
    fig.savefig(paths["plots"] / "regression_actual_vs_predicted.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    style_ax(ax, "Bandwidth Prediction Residuals")
    residuals = pred_table["pred_BW"] - pred_table["BW"]
    ax.hist(residuals, bins=22, color="#7c3aed", alpha=0.85, edgecolor="white")
    ax.axvline(0, color="#111827", linestyle="--", linewidth=1.3)
    ax.set_xlabel("Predicted BW - Actual BW (GHz)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(paths["plots"] / "bandwidth_residuals.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_feature_importance(clf: dict[str, object], feature_names: list[str], paths: dict[str, Path]) -> pd.DataFrame:
    rows = []
    for name, model in clf["base"]:                                                                                                     #type:ignore
        if hasattr(model, "feature_importances_"):
            rows.append(pd.Series(model.feature_importances_, index=feature_names, name=name))
    imp = pd.concat(rows, axis=1)
    imp["mean_importance"] = imp.mean(axis=1)
    out = imp.sort_values("mean_importance", ascending=False).reset_index(names="feature")
    out.to_csv(paths["tables"] / "feature_importance.csv", index=False)

    top = out.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    style_ax(ax, "Top 20 Feature Importances")
    ax.barh(top["feature"], top["mean_importance"], color="#0f766e")
    ax.set_xlabel("Mean tree importance")
    fig.tight_layout()
    fig.savefig(paths["plots"] / "feature_importance_top20.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def make_grid() -> pd.DataFrame:
    return pd.DataFrame(
        [{"LG": lg, "WR": wr, "LR": lr, "WF": wf} for lg, wr, lr, wf in product(LG_VALS, WR_VALS, LR_VALS, WF_VALS)]
    )


def rank_grid(
    df: pd.DataFrame,
    clf: dict[str, object],
    regs: dict[str, list[tuple[str, object]]],
    paths: dict[str, Path],
    lookup_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = make_grid()
    known = set(zip(lookup_table["LG"], lookup_table["WR"], lookup_table["LR"], lookup_table["WF"]))
    grid["known_in_data"] = [tuple(x) in known for x in zip(grid["LG"].round(3), grid["WR"].round(3), grid["LR"].round(3), grid["WF"].round(3))]
    Xg = build_features(grid)
    grid["prob_resonant"] = predict_res_prob(clf, Xg)
    grid["pred_BW"], grid["bw_model_std"] = predict_reg(regs, Xg, "BW")
    grid["pred_FL"], grid["fl_model_std"] = predict_reg(regs, Xg, "FL")
    grid["pred_FU"], grid["fu_model_std"] = predict_reg(regs, Xg, "FU")
    grid = attach_vlookup_predictions(grid, lookup_table)
    grid = apply_vlookup_to_predictions(grid)
    known_lookup = grid["known_in_lookup"].to_numpy(bool)
    grid["prob_resonant"] = np.where(known_lookup, grid["prob_resonant_lookup_assisted"], grid["prob_resonant"])
    grid["pred_BW"] = np.where(known_lookup, grid["pred_BW_lookup_assisted"], grid["pred_BW"])
    grid["pred_FL"] = np.where(known_lookup, grid["pred_FL_lookup_assisted"], grid["pred_FL"])
    grid["pred_FU"] = np.where(known_lookup, grid["pred_FU_lookup_assisted"], grid["pred_FU"])
    for col in ["bw_model_std", "fl_model_std", "fu_model_std"]:
        grid[col] = np.where(known_lookup, 0.0, grid[col])
    grid["pred_BW"] = grid["pred_BW"].clip(lower=0)
    grid["pred_fc"] = (grid["pred_FL"] + grid["pred_FU"]) / 2.0
    grid["pred_frac_bw"] = grid["pred_BW"] / grid["pred_fc"]
    grid["pred_in_4p5_9GHz"] = ((grid["pred_FL"] >= TARGET_FL_MIN) & (grid["pred_FU"] <= TARGET_FU_MAX)).astype(int)
    grid["expected_BW"] = grid["prob_resonant"] * grid["pred_BW"]
    grid["selection_score"] = grid["expected_BW"] * (0.75 + 0.25 * grid["pred_in_4p5_9GHz"]) / (1.0 + grid["bw_model_std"])
    top_all = grid.sort_values("selection_score", ascending=False).reset_index(drop=True)
    top_unseen = top_all[~top_all["known_in_data"]].reset_index(drop=True)
    top_all.head(100).to_csv(paths["tables"] / "top_100_all_grid_predictions.csv", index=False)
    top_unseen.head(100).to_csv(paths["tables"] / "top_100_unseen_grid_predictions.csv", index=False)
    return top_all, top_unseen


def physics_guided_candidates(
    df: pd.DataFrame, top_unseen: pd.DataFrame, paths: dict[str, Path], top_n: int
) -> pd.DataFrame:
    c_mm_per_ns = 300.0
    patch_length_mm = 11.0
    res = df[df["BW"] > 0].copy()
    res["eps_eff_est"] = (150.0 / (patch_length_mm * res["fc"])) ** 2
    eps_eff = float(np.clip(res["eps_eff_est"].replace([np.inf, -np.inf], np.nan).dropna().median(), 1.2, 12.0))
    res["lambda_g_mm"] = c_mm_per_ns / (res["fc"] * np.sqrt(eps_eff))
    res["WR_lambda_ratio"] = res["WR"] / res["lambda_g_mm"]
    res["LG_lambda_ratio"] = res["LG"] / res["lambda_g_mm"]
    res["WF_lambda_ratio"] = res["WF"] / res["lambda_g_mm"]
    high = res.sort_values("BW", ascending=False).head(max(30, int(0.05 * len(res))))
    targets = {
        "eps_eff_used": eps_eff,
        "high_bw_WR_lambda_ratio_median": float(high["WR_lambda_ratio"].median()),
        "high_bw_LG_lambda_ratio_median": float(high["LG_lambda_ratio"].median()),
        "high_bw_WF_lambda_ratio_median": float(high["WF_lambda_ratio"].median()),
    }

    ranked = top_unseen.copy()
    fc_grid = np.linspace(5.0, 8.5, 36)
    best_scores, best_fc = [], []
    for row in ranked.itertuples(index=False):
        scores = []
        for fc in fc_grid:
            lambda_g = c_mm_per_ns / (fc * np.sqrt(eps_eff))
            wr_ratio = row.WR / lambda_g
            lg_ratio = row.LG / lambda_g
            wf_ratio = row.WF / lambda_g
            score = np.exp(-((wr_ratio - targets["high_bw_WR_lambda_ratio_median"]) / 0.08) ** 2)
            score *= np.exp(-((lg_ratio - targets["high_bw_LG_lambda_ratio_median"]) / 0.07) ** 2)
            score *= np.exp(-((wf_ratio - targets["high_bw_WF_lambda_ratio_median"]) / 0.05) ** 2)
            scores.append(score)
        i = int(np.argmax(scores))
        best_scores.append(float(scores[i]))
        best_fc.append(float(fc_grid[i]))
    ranked["physics_fc"] = best_fc
    ranked["physics_score"] = best_scores
    ranked["physics_guided_score"] = ranked["selection_score"] * (0.70 + 0.30 * ranked["physics_score"])
    ranked = ranked.sort_values("physics_guided_score", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    ranked.head(top_n).to_csv(paths["tables"] / f"physics_guided_top_{top_n}_unseen_predictions.csv", index=False)
    (paths["root"] / "physics_calibration_summary.json").write_text(json.dumps(targets, indent=2))
    return ranked


def minmax(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def active_learning_candidates(
    top_unseen: pd.DataFrame,
    physics_ranked: pd.DataFrame,
    paths: dict[str, Path],
    active_n: int,
) -> pd.DataFrame:
    pool = top_unseen.copy()
    physics_cols = physics_ranked[
        ["LG", "WR", "LR", "WF", "physics_score", "physics_guided_score", "physics_fc"]
    ].copy()
    pool = pool.merge(physics_cols, on=["LG", "WR", "LR", "WF"], how="left")
    pool["classification_uncertainty"] = 1.0 - (2.0 * (pool["prob_resonant"] - 0.5).abs()).clip(upper=1.0)
    pool["total_model_std"] = pool[["bw_model_std", "fl_model_std", "fu_model_std"]].sum(axis=1)
    pool["active_learning_score"] = (
        0.35 * minmax(pool["classification_uncertainty"])
        + 0.25 * minmax(pool["total_model_std"])
        + 0.25 * minmax(pool["expected_BW"])
        + 0.15 * minmax(pool["physics_guided_score"].fillna(pool["selection_score"]))
    )
    pool["why_simulate"] = "balanced active-learning candidate"

    exploitation = pool.sort_values("expected_BW", ascending=False).head(max(1, active_n // 4)).copy()
    exploitation["why_simulate"] = "high predicted bandwidth"
    boundary = pool.sort_values("classification_uncertainty", ascending=False).head(max(1, active_n // 4)).copy()
    boundary["why_simulate"] = "resonant/dead boundary uncertainty"
    uncertainty = pool.sort_values("total_model_std", ascending=False).head(max(1, active_n // 4)).copy()
    uncertainty["why_simulate"] = "high regression disagreement"
    balanced = pool.sort_values("active_learning_score", ascending=False).head(active_n).copy()

    selected = pd.concat([exploitation, boundary, uncertainty, balanced], ignore_index=True)
    selected = (
        selected.sort_values("active_learning_score", ascending=False)
        .drop_duplicates(subset=["LG", "WR", "LR", "WF"], keep="first")
        .head(active_n)
        .reset_index(drop=True)
    )
    selected.insert(0, "active_rank", np.arange(1, len(selected) + 1))

    export_cols = [
        "active_rank",
        "LG",
        "WR",
        "LR",
        "WF",
        "why_simulate",
        "active_learning_score",
        "classification_uncertainty",
        "total_model_std",
        "prob_resonant",
        "pred_FL",
        "pred_FU",
        "pred_BW",
        "expected_BW",
        "bw_model_std",
        "fl_model_std",
        "fu_model_std",
        "physics_fc",
        "physics_score",
        "physics_guided_score",
    ]
    export = selected[[col for col in export_cols if col in selected.columns]]
    export.to_csv(paths["tables"] / f"active_learning_next_{active_n}_simulation_plan.csv", index=False)

    feedback_template = export.rename(
        columns={
            "pred_FL": "Pred Freq(Lower) GHz",
            "pred_FU": "Pred Freq(Upper) GHz",
            "pred_BW": "Pred BW GHz",
            "prob_resonant": "Probability resonant",
            "expected_BW": "Expected BW GHz",
            "active_learning_score": "Combined score",
        }
    )
    feedback_template["Known in old dataset"] = False
    feedback_template["Actual FL"] = ""
    feedback_template["Actual FU"] = ""
    feedback_template["Actual BW"] = ""
    feedback_cols = [
        "LG",
        "WR",
        "LR",
        "WF",
        "physics_fc",
        "Pred Freq(Lower) GHz",
        "Pred Freq(Upper) GHz",
        "Pred BW GHz",
        "Probability resonant",
        "Expected BW GHz",
        "Combined score",
        "Known in old dataset",
        "Actual FL",
        "Actual FU",
        "Actual BW",
    ]
    feedback_template[[col for col in feedback_cols if col in feedback_template.columns]].to_csv(
        paths["tables"] / f"active_learning_feedback_template_next_{active_n}.csv", index=False
    )
    return export


def save_candidate_plots(df: pd.DataFrame, top_unseen: pd.DataFrame, physics_ranked: pd.DataFrame, paths: dict[str, Path]) -> None:
    best_known = df.sort_values("BW", ascending=False).head(10).copy()
    best_unseen = top_unseen.head(10).copy()
    labels = ["Known simulated"] * len(best_known) + ["ML unseen"] * len(best_unseen)
    values = pd.concat([best_known["BW"].rename("BW"), best_unseen["pred_BW"].rename("BW")], ignore_index=True)
    x = np.arange(len(values))

    fig, ax = plt.subplots(figsize=(12, 5.2))
    style_ax(ax, "Known Best Designs vs Predicted Best Unseen Designs")
    colors = ["#16a34a" if label == "Known simulated" else "#2563eb" for label in labels]
    ax.bar(x, values, color=colors)
    ax.set_xticks(x, [f"K{i+1}" if i < len(best_known) else f"U{i-len(best_known)+1}" for i in x], rotation=0)
    ax.set_ylabel("Bandwidth (GHz)")
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color="#16a34a", label="Known simulated"),                                                                  #type:ignore
            plt.Rectangle((0, 0), 1, 1, color="#2563eb", label="Predicted unseen"),                                                                 #type:ignore
        ]
    )
    fig.tight_layout()
    fig.savefig(paths["plots"] / "known_vs_predicted_best_bandwidth.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    top = physics_ranked.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 7))
    style_ax(ax, "Top Physics-Guided Unseen Candidates")
    labels = [f"L{r.LG:g} W{r.WR:g} R{r.LR:g} F{r.WF:g}" for r in top.itertuples()]
    ax.barh(labels, top["pred_BW"], color="#7c3aed")
    ax.set_xlabel("Predicted BW (GHz)")
    fig.tight_layout()
    fig.savefig(paths["plots"] / "top_physics_guided_candidates.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_probability_plot(y_true: np.ndarray, prob: np.ndarray, threshold: float, paths: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    style_ax(ax, "Validation Probability Separation")
    ax.hist(prob[y_true == 0], bins=22, alpha=0.75, color="#dc2626", label="Dead")
    ax.hist(prob[y_true == 1], bins=22, alpha=0.75, color="#16a34a", label="Resonant")
    ax.axvline(threshold, color="#111827", linestyle="--", linewidth=1.4, label=f"Threshold {threshold:.2f}")
    ax.set_xlabel("Predicted resonant probability")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths["plots"] / "validation_probability_distribution.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> None:
    paths = ensure_dirs(Path(args.output))
    data_path = resolve_data_path(args.data)
    df = load_data(data_path)
    original_rows = len(df)
    df, feedback_rows, feedback_path = combine_with_feedback(df, args.feedback)
    print(f"Loaded {len(df)} unique rows from {data_path}")
    if feedback_path:
        print(f"Included {feedback_rows} feedback simulation row(s) from {feedback_path}")
    print(f"Resonant rows: {int(df.is_resonant.sum())}; dead rows: {int((df.is_resonant == 0).sum())}")
    lookup_table = build_vlookup_table(df)
    lookup_table.to_csv(paths["tables"] / "known_data_vlookup_table.csv", index=False)
    vlookup_known_classifier, vlookup_known_regression = evaluate_vlookup_known_data(lookup_table)
    pd.DataFrame([vlookup_known_classifier]).to_csv(paths["tables"] / "known_data_vlookup_classifier_metrics.csv", index=False)
    pd.DataFrame.from_dict(vlookup_known_regression, orient="index").reset_index(names="target").to_csv(
        paths["tables"] / "known_data_vlookup_regression_metrics.csv", index=False
    )

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=df["is_resonant"],
    )
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    X_train, X_val = build_features(df_train), build_features(df_val)
    y_train, y_val = df_train["is_resonant"].to_numpy(), df_val["is_resonant"].to_numpy()

    epoch_history, monitor = train_epoch_monitor(df_train, df_val, args.epochs, paths)
    save_epoch_plots(epoch_history, paths)
    save_confusion_plot(monitor["y_true"], monitor["y_pred"], "MLP Epoch Monitor Confusion Matrix", paths["plots"] / "epoch_monitor_confusion_matrix.png")                                      #type:ignore

    clf = fit_classifier(X_train, y_train)
    val_prob = predict_res_prob(clf, X_val)
    val_pred = (val_prob >= clf["threshold"]).astype(int)
    val_metrics = classifier_metrics(y_val, val_prob, clf["threshold"])                                                                                                                     #type:ignore
    save_confusion_plot(y_val, val_pred, "Consolidated Model Validation Confusion Matrix", paths["plots"] / "consolidated_confusion_matrix.png")
    save_probability_plot(y_val, val_prob, clf["threshold"], paths)                                                                                                                         #type:ignore

    technique_metrics, _ = technique_comparison(clf, X_val, y_val)
    technique_metrics.to_csv(paths["tables"] / "technique_comparison_metrics.csv", index=False)
    save_technique_plots(technique_metrics, paths)

    regs, reg_train_data = fit_regressors(df_train)
    reg_metrics, pred_table = evaluate_regressors(regs, df_val)
    pred_table.to_csv(paths["tables"] / "validation_regression_predictions.csv", index=False)
    vlookup_val_classifier, vlookup_val_regression, vlookup_val_predictions = evaluate_lookup_assisted_validation(
        df_val, val_prob, pred_table, lookup_table, clf["threshold"]                                                                                                                            #type:ignore
    )
    vlookup_val_predictions.to_csv(paths["tables"] / "validation_vlookup_assisted_predictions.csv", index=False)
    pd.DataFrame([vlookup_val_classifier]).to_csv(
        paths["tables"] / "validation_vlookup_assisted_classifier_metrics.csv", index=False
    )
    pd.DataFrame.from_dict(vlookup_val_regression, orient="index").reset_index(names="target").to_csv(
        paths["tables"] / "validation_vlookup_assisted_regression_metrics.csv", index=False
    )
    save_regression_plots(pred_table, paths)
    feature_importance = save_feature_importance(clf, X_train.columns.tolist(), paths)

    final_clf = fit_classifier(build_features(df), df["is_resonant"].to_numpy())
    final_regs, final_reg_train_data = fit_regressors(df)
    top_all, top_unseen = rank_grid(df, final_clf, final_regs, paths, lookup_table)
    physics_ranked = physics_guided_candidates(df, top_unseen, paths, args.top_n)
    active_learning_plan = active_learning_candidates(top_unseen, physics_ranked, paths, args.active_n)
    save_candidate_plots(df, top_unseen, physics_ranked, paths)

    artifact = {
        "classifier": final_clf,
        "regressors": final_regs,
        "feature_columns": X_train.columns.tolist(),
        "threshold": final_clf["threshold"],
        "vlookup_table": lookup_table,
        "vlookup_key_columns": ["LG", "WR", "LR", "WF"],
        "data_path": str(data_path),
        "target_window": {"FL_min": TARGET_FL_MIN, "FU_max": TARGET_FU_MAX},
    }
    with open(paths["models"] / "consolidated_antenna_model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    summary = {
        "data_path": str(data_path),
        "feedback_path": str(feedback_path) if feedback_path else None,
        "original_rows_before_feedback": int(original_rows),
        "feedback_rows_loaded": int(feedback_rows),
        "rows": int(len(df)),
        "features": int(X_train.shape[1]),
        "train_rows": int(len(df_train)),
        "validation_rows": int(len(df_val)),
        "resonant_rows": int(df["is_resonant"].sum()),
        "dead_rows": int((df["is_resonant"] == 0).sum()),
        "target_window_rows": int(df["in_target_window"].sum()),
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().to_dict().items()},
        "validation_classifier": val_metrics,
        "validation_regression": reg_metrics,
        "vlookup_known_data_classifier": vlookup_known_classifier,
        "vlookup_known_data_regression": vlookup_known_regression,
        "vlookup_assisted_validation_classifier": vlookup_val_classifier,
        "vlookup_assisted_validation_regression": vlookup_val_regression,
        "vlookup_rows": int(len(lookup_table)),
        "vlookup_validation_coverage": float(vlookup_val_predictions["known_in_lookup"].mean()),
        "chosen_threshold": float(clf["threshold"]),                                                                                                                #type:ignore
        "best_known_data_row": df.sort_values("BW", ascending=False).iloc[0][["LG", "WR", "LR", "WF", "FL", "FU", "BW"]].to_dict(),
        "best_unseen_ml_prediction": top_unseen.iloc[0][
            ["LG", "WR", "LR", "WF", "prob_resonant", "pred_FL", "pred_FU", "pred_BW", "selection_score"]
        ].to_dict(),
        "best_physics_guided_prediction": physics_ranked.iloc[0][
            ["LG", "WR", "LR", "WF", "prob_resonant", "pred_FL", "pred_FU", "pred_BW", "physics_score", "physics_guided_score"]
        ].to_dict(),
        "best_active_learning_candidate": active_learning_plan.iloc[0].to_dict(),
        "top_features": feature_importance.head(10)[["feature", "mean_importance"]].to_dict(orient="records"),
        "regression_training_rows": int(len(reg_train_data)),
        "final_regression_training_rows": int(len(final_reg_train_data)),
    }
    (paths["root"] / "research_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nValidation classifier metrics")
    print(json.dumps(val_metrics, indent=2))
    print("\nValidation regression metrics")
    print(json.dumps(reg_metrics, indent=2))
    print("\nVLOOKUP-assisted validation classifier metrics")
    print(json.dumps(vlookup_val_classifier, indent=2))
    print(f"\nSaved outputs under: {paths['root'].resolve()}")
    print(f"Main plots folder: {paths['plots'].resolve()}")
    print(f"Top physics-guided candidates: {paths['tables'] / f'physics_guided_top_{args.top_n}_unseen_predictions.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-file antenna research pipeline.")
    parser.add_argument("--data", default=None, help="Path to UBW-data-final.xlsx. Defaults to data/UBW-data-final.xlsx.")
    parser.add_argument(
        "--feedback",
        default=None,
        help="Optional CSV with simulator feedback. Defaults to data/feedback_simulations.csv when that file exists.",
    )
    parser.add_argument("--output", default="research_outputs", help="Directory for plots, tables, summaries, and model files.")
    parser.add_argument("--epochs", type=int, default=20, help="Epochs for the MLP learning-curve monitor.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Validation fraction.")
    parser.add_argument("--top-n", type=int, default=25, help="Number of physics-guided unseen candidates to export.")
    parser.add_argument("--active-n", type=int, default=20, help="Number of active-learning simulation candidates to export.")
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
