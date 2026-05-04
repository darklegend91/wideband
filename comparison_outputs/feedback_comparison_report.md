# Feedback Comparison Report

## Main Metric Changes

- Data - rows: baseline `2039`, after feedback `2063`, delta `24` (Training coverage increased).
- Data - resonant_rows: baseline `1615`, after feedback `1637`, delta `22` (Training coverage increased).
- Data - dead_rows: baseline `424`, after feedback `426`, delta `2` (Training coverage increased).
- Data - target_window_rows: baseline `1615`, after feedback `1637`, delta `22` (Training coverage increased).
- Classification - Accuracy: baseline `0.882353`, after feedback `0.891041`, delta `0.00868822` (Improved).
- Classification - Balanced accuracy: baseline `0.8`, after feedback `0.78759`, delta `-0.0124103` (Reduced).
- Classification - Precision: baseline `0.912913`, after feedback `0.905444`, delta `-0.00746879` (Reduced).
- Classification - Recall: baseline `0.941176`, after feedback `0.963415`, delta `0.0222382` (Improved).
- Classification - F1 score: baseline `0.926829`, after feedback `0.93353`, delta `0.00670101` (Improved).
- Classification - AUC-ROC: baseline `0.905154`, after feedback `0.919907`, delta `0.0147529` (Improved).
- Classification - Log loss: baseline `0.520627`, after feedback `0.671291`, delta `0.150664` (Reduced).
- Regression - BW R2: baseline `0.556715`, after feedback `0.562489`, delta `0.00577403` (Improved).
- Regression - BW MAE: baseline `0.334808`, after feedback `0.329881`, delta `-0.00492725` (Improved).
- Regression - BW RMSE: baseline `0.522644`, after feedback `0.546273`, delta `0.0236291` (Reduced).
- Regression - FL R2: baseline `0.670409`, after feedback `0.743055`, delta `0.0726462` (Improved).
- Regression - FL MAE: baseline `0.230796`, after feedback `0.217434`, delta `-0.0133611` (Improved).
- Regression - FL RMSE: baseline `0.479623`, after feedback `0.409585`, delta `-0.0700378` (Improved).
- Regression - FU R2: baseline `0.570347`, after feedback `0.56665`, delta `-0.00369717` (Reduced).
- Regression - FU MAE: baseline `0.425194`, after feedback `0.407924`, delta `-0.0172694` (Improved).
- Regression - FU RMSE: baseline `0.627208`, after feedback `0.626635`, delta `-0.000573272` (Improved).

## Feedback Simulation Error

- FL: original prediction MAE on feedback points = `0.5395 GHz`, mean signed error = `0.4650 GHz`.
- FU: original prediction MAE on feedback points = `1.3506 GHz`, mean signed error = `1.0117 GHz`.
- BW: original prediction MAE on feedback points = `0.8452 GHz`, mean signed error = `0.5263 GHz`.

## Candidate Ranking Change

- Baseline top candidate: `5.5_12.25_3.0_3.5`, predicted BW `3.2373 GHz`.
- After-feedback top candidate: `6.0_12.25_3.3_4.5`, predicted BW `3.4006 GHz`.
- Top-25 overlap count: `5` designs.

## Technique-Level Notes

- Extra Trees: accuracy delta `0.0064`, F1 delta `0.0046`, AUC delta `0.0139`.
- Random Forest: accuracy delta `0.0111`, F1 delta `0.0078`, AUC delta `0.0158`.
- Hist Gradient Boosting: accuracy delta `0.0211`, F1 delta `0.0139`, AUC delta `0.0135`.
- ET + RF Average: accuracy delta `0.0160`, F1 delta `0.0102`, AUC delta `0.0159`.
- ET + RF + HGB Average: accuracy delta `0.0257`, F1 delta `0.0166`, AUC delta `0.0149`.
- Consolidated Stacked Model: accuracy delta `0.0087`, F1 delta `0.0067`, AUC delta `0.0148`.

## Generated Plots

- `plots/classification_before_after.png`
- `plots/regression_r2_before_after.png`
- `plots/regression_mae_before_after.png`
- `plots/regression_rmse_before_after.png`
- `plots/technique_before_after.png`
- `plots/feedback_predicted_vs_actual.png`
- `plots/feedback_error_summary.png`
- `plots/top_candidate_before_after.png`

## Interpretation For Paper

The feedback loop adds real EM simulation results from model-recommended designs back into the training set. This changes both the learned response surface and the candidate ranking. The comparison shows whether the feedback improves classification discrimination, regression fit for FL/FU/BW, and practical design selection.