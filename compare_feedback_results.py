"""
Compare baseline model results against retraining with simulator feedback.

Inputs:
  research_outputs/
  research_outputs_feedback/
  data/feedback_simulations.csv

Outputs:
  comparison_outputs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PLOT_DPI = 180


def load_summary(path: Path) -> dict:
    return json.loads((path / "research_summary.json").read_text())


def style_ax(ax, title: str | None = None) -> None:
    ax.set_facecolor("#f8fafc")
    ax.grid(True, alpha=0.25, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def pct(value: float) -> float:
    return 100.0 * value


def metric_delta_rows(base: dict, feedback: dict) -> pd.DataFrame:
    rows = []
    for key in ["rows", "resonant_rows", "dead_rows", "target_window_rows"]:
        rows.append(
            {
                "aspect": "Data",
                "metric": key,
                "baseline": base.get(key),
                "after_feedback": feedback.get(key),
                "delta": feedback.get(key, 0) - base.get(key, 0),
                "interpretation": "Training coverage increased" if feedback.get(key, 0) >= base.get(key, 0) else "Training coverage decreased",
            }
        )

    for key, label, higher_better in [
        ("accuracy", "Accuracy", True),
        ("balanced_accuracy", "Balanced accuracy", True),
        ("precision", "Precision", True),
        ("recall", "Recall", True),
        ("f1", "F1 score", True),
        ("auc", "AUC-ROC", True),
        ("loss", "Log loss", False),
    ]:
        b = base["validation_classifier"][key]
        f = feedback["validation_classifier"][key]
        improved = f > b if higher_better else f < b
        rows.append(
            {
                "aspect": "Classification",
                "metric": label,
                "baseline": b,
                "after_feedback": f,
                "delta": f - b,
                "interpretation": "Improved" if improved else "Reduced",
            }
        )

    for target in ["BW", "FL", "FU"]:
        for key, label, higher_better in [
            ("r2", f"{target} R2", True),
            ("mae", f"{target} MAE", False),
            ("rmse", f"{target} RMSE", False),
        ]:
            b = base["validation_regression"][target][key]
            f = feedback["validation_regression"][target][key]
            improved = f > b if higher_better else f < b
            rows.append(
                {
                    "aspect": "Regression",
                    "metric": label,
                    "baseline": b,
                    "after_feedback": f,
                    "delta": f - b,
                    "interpretation": "Improved" if improved else "Reduced",
                }
            )
    return pd.DataFrame(rows)


def compare_techniques(base_dir: Path, feedback_dir: Path) -> pd.DataFrame:
    base = pd.read_csv(base_dir / "tables" / "technique_comparison_metrics.csv")
    feedback = pd.read_csv(feedback_dir / "tables" / "technique_comparison_metrics.csv")
    merged = base.merge(feedback, on="technique", suffixes=("_baseline", "_after_feedback"))
    for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "loss"]:
        merged[f"{metric}_delta"] = merged[f"{metric}_after_feedback"] - merged[f"{metric}_baseline"]
    return merged


def compare_candidates(base_dir: Path, feedback_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(base_dir / "tables" / "physics_guided_top_25_unseen_predictions.csv")
    feedback = pd.read_csv(feedback_dir / "tables" / "physics_guided_top_25_unseen_predictions.csv")
    key_cols = ["LG", "WR", "LR", "WF"]
    base["design_key"] = base[key_cols].astype(str).agg("_".join, axis=1)
    feedback["design_key"] = feedback[key_cols].astype(str).agg("_".join, axis=1)
    overlap = sorted(set(base["design_key"]) & set(feedback["design_key"]))
    overlap_df = (
        base[["design_key", "rank", "pred_BW", "physics_guided_score"]]
        .merge(
            feedback[["design_key", "rank", "pred_BW", "physics_guided_score"]],
            on="design_key",
            suffixes=("_baseline", "_after_feedback"),
        )
        .sort_values("rank_after_feedback")
    )
    top_summary = pd.DataFrame(
        {
            "scenario": ["baseline", "after_feedback"],
            "top1_design": [base.iloc[0]["design_key"], feedback.iloc[0]["design_key"]],
            "top1_pred_BW": [base.iloc[0]["pred_BW"], feedback.iloc[0]["pred_BW"]],
            "top1_score": [base.iloc[0]["physics_guided_score"], feedback.iloc[0]["physics_guided_score"]],
            "top25_overlap_count": [len(overlap), len(overlap)],
        }
    )
    return overlap_df, top_summary


def feedback_error_table(feedback_csv: Path) -> pd.DataFrame:
    raw = pd.read_csv(feedback_csv)
    if len(raw.columns) == 1 and raw.columns[0].strip().lower() not in {"lg", "wr", "lr", "wf"}:
        raw = pd.read_csv(feedback_csv, skiprows=1)
    raw.columns = [str(col).strip() for col in raw.columns]
    rename = {
        "Pred Freq(Lower) GHz": "pred_FL",
        "Pred Freq(Upper) GHz": "pred_FU",
        "Pred BW GHz": "pred_BW",
        "Probability resonant": "pred_prob_resonant",
        "Expected BW GHz": "pred_expected_BW",
        "Combined score": "pred_combined_score",
        "Actual FL": "actual_FL",
        "Actual FU": "actual_FU",
        "Actual BW": "actual_BW",
    }
    df = raw.rename(columns=rename).copy()
    needed = ["LG", "WR", "LR", "WF", "pred_FL", "pred_FU", "pred_BW", "actual_FL", "actual_FU", "actual_BW"]
    df = df.dropna(subset=needed).copy()
    for col in needed:
        df[col] = df[col].astype(float)
    for target in ["FL", "FU", "BW"]:
        df[f"{target}_error"] = df[f"pred_{target}"] - df[f"actual_{target}"]
        df[f"{target}_abs_error"] = df[f"{target}_error"].abs()
    df["design_key"] = df[["LG", "WR", "LR", "WF"]].astype(str).agg("_".join, axis=1)
    return df


def save_metric_plots(metric_df: pd.DataFrame, out_dir: Path) -> None:
    clf = metric_df[metric_df["aspect"] == "Classification"].copy()
    fig, ax = plt.subplots(figsize=(11, 5.2))
    style_ax(ax, "Classification Metrics Before and After Feedback")
    x = np.arange(len(clf))
    width = 0.36
    ax.bar(x - width / 2, clf["baseline"], width, label="Baseline", color="#2563eb")
    ax.bar(x + width / 2, clf["after_feedback"], width, label="After feedback", color="#16a34a")
    ax.set_xticks(x, clf["metric"], rotation=25, ha="right")
    ax.set_ylim(0, max(1.05, float(clf[["baseline", "after_feedback"]].max().max()) * 1.15))
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "classification_before_after.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    reg = metric_df[metric_df["aspect"] == "Regression"].copy()
    for suffix, title, ylabel in [
        ("R2", "Regression R2 Before and After Feedback", "R2"),
        ("MAE", "Regression MAE Before and After Feedback", "GHz"),
        ("RMSE", "Regression RMSE Before and After Feedback", "GHz"),
    ]:
        subset = reg[reg["metric"].str.endswith(suffix)].copy()
        fig, ax = plt.subplots(figsize=(8.5, 5))
        style_ax(ax, title)
        x = np.arange(len(subset))
        ax.bar(x - width / 2, subset["baseline"], width, label="Baseline", color="#2563eb")
        ax.bar(x + width / 2, subset["after_feedback"], width, label="After feedback", color="#16a34a")
        ax.set_xticks(x, subset["metric"].str.replace(f" {suffix}", "", regex=False))
        ax.set_ylabel(ylabel)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"regression_{suffix.lower()}_before_after.png", dpi=PLOT_DPI, bbox_inches="tight")
        plt.close(fig)


def save_technique_plot(technique_df: pd.DataFrame, out_dir: Path) -> None:
    labels = technique_df["technique"].str.replace("Consolidated Stacked Model", "Consolidated\nStacked").str.replace(
        "Hist Gradient Boosting", "HistGB"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, metric, title in [
        (axes[0], "accuracy", "Technique Accuracy Before/After Feedback"),
        (axes[1], "f1", "Technique F1 Before/After Feedback"),
    ]:
        style_ax(ax, title)
        x = np.arange(len(technique_df))
        width = 0.36
        ax.bar(x - width / 2, technique_df[f"{metric}_baseline"], width, label="Baseline", color="#2563eb")
        ax.bar(x + width / 2, technique_df[f"{metric}_after_feedback"], width, label="After feedback", color="#16a34a")
        ax.set_xticks(x, labels, rotation=22, ha="right")
        ax.set_ylim(0, 1.05)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "technique_before_after.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_feedback_error_plots(errors: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, target in zip(axes, ["FL", "FU", "BW"]):
        style_ax(ax, f"Feedback Points: Predicted vs Actual {target}")
        ax.scatter(errors[f"actual_{target}"], errors[f"pred_{target}"], s=38, color="#7c3aed", alpha=0.8)
        lo = min(errors[f"actual_{target}"].min(), errors[f"pred_{target}"].min())
        hi = max(errors[f"actual_{target}"].max(), errors[f"pred_{target}"].max())
        ax.plot([lo, hi], [lo, hi], "--", color="#111827")
        ax.set_xlabel(f"Actual {target} (GHz)")
        ax.set_ylabel(f"Original predicted {target} (GHz)")
    fig.tight_layout()
    fig.savefig(out_dir / "feedback_predicted_vs_actual.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    mae = pd.DataFrame(
        {
            "target": ["FL", "FU", "BW"],
            "mae": [errors["FL_abs_error"].mean(), errors["FU_abs_error"].mean(), errors["BW_abs_error"].mean()],
            "bias": [errors["FL_error"].mean(), errors["FU_error"].mean(), errors["BW_error"].mean()],
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    style_ax(ax, "Original Model Error on Feedback Simulations")
    x = np.arange(len(mae))
    width = 0.36
    ax.bar(x - width / 2, mae["mae"], width, label="MAE", color="#dc2626")
    ax.bar(x + width / 2, mae["bias"], width, label="Mean signed error", color="#ca8a04")
    ax.axhline(0, color="#111827", linewidth=1)
    ax.set_xticks(x, mae["target"])
    ax.set_ylabel("GHz")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "feedback_error_summary.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def save_candidate_plot(top_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.8))
    style_ax(ax, "Top Recommended Candidate Before and After Feedback")
    x = np.arange(len(top_summary))
    ax.bar(x, top_summary["top1_pred_BW"], color=["#2563eb", "#16a34a"])
    ax.set_xticks(x, top_summary["scenario"])
    ax.set_ylabel("Top-1 predicted BW (GHz)")
    for i, row in top_summary.iterrows():
        ax.text(i, row["top1_pred_BW"] + 0.04, row["top1_design"], ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "top_candidate_before_after.png", dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def write_markdown(
    metric_df: pd.DataFrame,
    technique_df: pd.DataFrame,
    overlap_df: pd.DataFrame,
    top_summary: pd.DataFrame,
    errors: pd.DataFrame,
    out_dir: Path,
) -> None:
    lines = ["# Feedback Comparison Report", ""]
    lines.append("## Main Metric Changes")
    lines.append("")
    for _, row in metric_df.iterrows():
        lines.append(
            f"- {row['aspect']} - {row['metric']}: baseline `{row['baseline']:.6g}`, "
            f"after feedback `{row['after_feedback']:.6g}`, delta `{row['delta']:.6g}` ({row['interpretation']})."
        )
    lines.append("")
    lines.append("## Feedback Simulation Error")
    lines.append("")
    for target in ["FL", "FU", "BW"]:
        lines.append(
            f"- {target}: original prediction MAE on feedback points = "
            f"`{errors[f'{target}_abs_error'].mean():.4f} GHz`, mean signed error = "
            f"`{errors[f'{target}_error'].mean():.4f} GHz`."
        )
    lines.append("")
    lines.append("## Candidate Ranking Change")
    lines.append("")
    lines.append(
        f"- Baseline top candidate: `{top_summary.iloc[0]['top1_design']}`, "
        f"predicted BW `{top_summary.iloc[0]['top1_pred_BW']:.4f} GHz`."
    )
    lines.append(
        f"- After-feedback top candidate: `{top_summary.iloc[1]['top1_design']}`, "
        f"predicted BW `{top_summary.iloc[1]['top1_pred_BW']:.4f} GHz`."
    )
    lines.append(f"- Top-25 overlap count: `{int(top_summary.iloc[0]['top25_overlap_count'])}` designs.")
    lines.append("")
    lines.append("## Technique-Level Notes")
    lines.append("")
    for _, row in technique_df.iterrows():
        lines.append(
            f"- {row['technique']}: accuracy delta `{row['accuracy_delta']:.4f}`, "
            f"F1 delta `{row['f1_delta']:.4f}`, AUC delta `{row['auc_delta']:.4f}`."
        )
    lines.append("")
    lines.append("## Generated Plots")
    lines.append("")
    for name in [
        "classification_before_after.png",
        "regression_r2_before_after.png",
        "regression_mae_before_after.png",
        "regression_rmse_before_after.png",
        "technique_before_after.png",
        "feedback_predicted_vs_actual.png",
        "feedback_error_summary.png",
        "top_candidate_before_after.png",
    ]:
        lines.append(f"- `plots/{name}`")
    lines.append("")
    lines.append("## Interpretation For Paper")
    lines.append("")
    lines.append(
        "The feedback loop adds real EM simulation results from model-recommended designs back into the training set. "
        "This changes both the learned response surface and the candidate ranking. The comparison shows whether the "
        "feedback improves classification discrimination, regression fit for FL/FU/BW, and practical design selection."
    )
    (out_dir / "feedback_comparison_report.md").write_text("\n".join(lines))


def run(args: argparse.Namespace) -> None:
    baseline_dir = Path(args.baseline)
    feedback_dir = Path(args.feedback)
    feedback_csv = Path(args.feedback_csv)
    out_dir = Path(args.output)
    plots_dir = out_dir / "plots"
    tables_dir = out_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    base_summary = load_summary(baseline_dir)
    feedback_summary = load_summary(feedback_dir)
    metric_df = metric_delta_rows(base_summary, feedback_summary)
    technique_df = compare_techniques(baseline_dir, feedback_dir)
    overlap_df, top_summary = compare_candidates(baseline_dir, feedback_dir)
    errors = feedback_error_table(feedback_csv)

    metric_df.to_csv(tables_dir / "metric_before_after_comparison.csv", index=False)
    technique_df.to_csv(tables_dir / "technique_before_after_comparison.csv", index=False)
    overlap_df.to_csv(tables_dir / "top25_candidate_overlap.csv", index=False)
    top_summary.to_csv(tables_dir / "top_candidate_summary.csv", index=False)
    errors.to_csv(tables_dir / "feedback_prediction_errors.csv", index=False)

    save_metric_plots(metric_df, plots_dir)
    save_technique_plot(technique_df, plots_dir)
    save_feedback_error_plots(errors, plots_dir)
    save_candidate_plot(top_summary, plots_dir)
    write_markdown(metric_df, technique_df, overlap_df, top_summary, errors, out_dir)

    print(f"Comparison report: {out_dir / 'feedback_comparison_report.md'}")
    print(f"Comparison plots:  {plots_dir}")
    print(f"Comparison tables: {tables_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and feedback-trained research outputs.")
    parser.add_argument("--baseline", default="research_outputs", help="Baseline output directory.")
    parser.add_argument("--feedback", default="research_outputs_feedback", help="Feedback-trained output directory.")
    parser.add_argument("--feedback-csv", default="data/feedback_simulations.csv", help="Feedback simulation CSV.")
    parser.add_argument("--output", default="comparison_outputs", help="Comparison output directory.")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
