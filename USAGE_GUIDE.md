# Wideband Antenna Model Usage Guide

This guide explains the project structure and the correct order for running the baseline model, feedback model, VLOOKUP-assisted model, and active-learning feedback loop.

## What This Project Does

The model learns from antenna simulation data:

```text
Inputs:  LG, WR, LR, WF
Outputs: FL, FU, BW
Label:   resonant if BW > 0, dead if BW = 0
```

It produces:

- resonant/dead classification
- lower frequency prediction, `FL`
- upper frequency prediction, `FU`
- bandwidth prediction, `BW`
- ranked unseen antenna candidates
- VLOOKUP exact predictions for already simulated points
- active-learning candidates for the next simulation batch

## Main Folder Structure

```text
final-files/
├── research_pipeline.py
├── compare_feedback_results.py
├── README.md
├── USAGE_GUIDE.md
├── MODEL_TECHNIQUES_EXPLAINED.md
├── requirements.txt
├── data/
│   ├── UBW-data-final.xlsx
│   └── feedback_simulations.csv
├── research_outputs/
├── research_outputs_feedback/
├── research_outputs_feedback_vlookup/
├── comparison_outputs/
├── code/
├── workflow/
├── model/
└── predictions/
```

## Important Files

| Path | Purpose |
|---|---|
| `research_pipeline.py` | Main script. Use this for training, feedback, VLOOKUP, candidate ranking, and active learning. |
| `compare_feedback_results.py` | Compares baseline outputs vs feedback-trained outputs. |
| `data/UBW-data-final.xlsx` | Original simulation dataset. |
| `data/feedback_simulations.csv` | Real simulation feedback rows added after testing model-recommended points. |
| `research_outputs/` | Baseline model outputs without feedback. |
| `research_outputs_feedback/` | Feedback-trained model outputs without the newer VLOOKUP output folder name. |
| `research_outputs_feedback_vlookup/` | Recommended current output folder for feedback + VLOOKUP + active learning. |
| `MODEL_TECHNIQUES_EXPLAINED.md` | Explanation of the ML techniques used. |

## Environment Setup

From this folder:

```bash
cd /Users/adityapathania/Codes/curin/antena-model/Wideband/final-files
```

Use the existing environment from the parent folder:

```bash
../wide-venv/bin/python --version
```

If you ever need a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then replace `../wide-venv/bin/python` in the commands below with:

```bash
python
```

## Recommended Run Order

Use this order when you want a clean experiment.

### 1. Run Baseline Model

This trains only on the original Excel dataset.

```bash
cd /Users/adityapathania/Codes/curin/antena-model/Wideband/final-files
../wide-venv/bin/python research_pipeline.py --output research_outputs
```

Main output:

```text
research_outputs/research_summary.json
research_outputs/models/consolidated_antenna_model.pkl
research_outputs/tables/top_100_unseen_grid_predictions.csv
research_outputs/tables/physics_guided_top_25_unseen_predictions.csv
```

Use this when you want the original model accuracy before adding feedback.

### 2. Run Feedback-Trained Model

This trains on original data plus real simulation feedback from `data/feedback_simulations.csv`.

```bash
../wide-venv/bin/python research_pipeline.py \
  --feedback data/feedback_simulations.csv \
  --output research_outputs_feedback
```

Main output:

```text
research_outputs_feedback/research_summary.json
research_outputs_feedback/tables/top_100_unseen_grid_predictions.csv
research_outputs_feedback/tables/physics_guided_top_25_unseen_predictions.csv
```

Use this to measure whether feedback improved the pure ML model.

### 3. Run Feedback + VLOOKUP + Active Learning

This is the current recommended command.

```bash
../wide-venv/bin/python research_pipeline.py \
  --feedback data/feedback_simulations.csv \
  --output research_outputs_feedback_vlookup \
  --active-n 20
```

For a faster test run while editing code:

```bash
../wide-venv/bin/python research_pipeline.py \
  --feedback data/feedback_simulations.csv \
  --output research_outputs_feedback_vlookup \
  --epochs 5 \
  --active-n 20
```

Main output:

```text
research_outputs_feedback_vlookup/research_summary.json
research_outputs_feedback_vlookup/models/consolidated_antenna_model.pkl
research_outputs_feedback_vlookup/tables/known_data_vlookup_table.csv
research_outputs_feedback_vlookup/tables/validation_vlookup_assisted_predictions.csv
research_outputs_feedback_vlookup/tables/active_learning_next_20_simulation_plan.csv
research_outputs_feedback_vlookup/tables/active_learning_feedback_template_next_20.csv
```

## VLOOKUP Mode

The VLOOKUP layer stores every known simulated antenna geometry:

```text
LG, WR, LR, WF -> FL, FU, BW, resonant/dead
```

If a prediction point already exists in old data or feedback data, the model returns the exact known value from the lookup table.

If the point is new, it falls back to ML prediction.

Important VLOOKUP files:

```text
research_outputs_feedback_vlookup/tables/known_data_vlookup_table.csv
research_outputs_feedback_vlookup/tables/known_data_vlookup_classifier_metrics.csv
research_outputs_feedback_vlookup/tables/known_data_vlookup_regression_metrics.csv
research_outputs_feedback_vlookup/tables/validation_vlookup_assisted_classifier_metrics.csv
research_outputs_feedback_vlookup/tables/validation_vlookup_assisted_regression_metrics.csv
```

Check VLOOKUP-assisted accuracy:

```bash
cat research_outputs_feedback_vlookup/tables/validation_vlookup_assisted_classifier_metrics.csv
```

Expected known-data behavior:

```text
accuracy = 1.0
balanced_accuracy = 1.0
f1 = 1.0
auc = 1.0
```

Important: this is exact memory for already simulated points. It is not the same as pure ML accuracy on unseen designs.

## Pure ML Accuracy

Check pure ML validation metrics here:

```bash
cat research_outputs_feedback_vlookup/research_summary.json
```

Look for:

```text
validation_classifier
validation_regression
```

These are the honest ML metrics without using the lookup answer.

Current example:

```text
validation_classifier.accuracy = about 0.891
vlookup_assisted_validation_classifier.accuracy = 1.0
```

Use pure ML metrics when discussing generalization. Use VLOOKUP metrics when discussing exact reuse of older simulated points.

## Active Learning Workflow

Active learning tells you which new antenna points to simulate next.

Run:

```bash
../wide-venv/bin/python research_pipeline.py \
  --feedback data/feedback_simulations.csv \
  --output research_outputs_feedback_vlookup \
  --active-n 20
```

Open:

```text
research_outputs_feedback_vlookup/tables/active_learning_next_20_simulation_plan.csv
```

This file ranks candidates using:

- high predicted bandwidth
- resonant/dead boundary uncertainty
- disagreement between regressors
- physics-guided score

Then open:

```text
research_outputs_feedback_vlookup/tables/active_learning_feedback_template_next_20.csv
```

Simulate those rows in HFSS/CST and fill:

```text
Actual FL
Actual FU
Actual BW
```

After filling the actual values, append the completed rows to:

```text
data/feedback_simulations.csv
```

Then rerun:

```bash
../wide-venv/bin/python research_pipeline.py \
  --feedback data/feedback_simulations.csv \
  --output research_outputs_feedback_vlookup \
  --active-n 20
```

Repeat this loop:

```text
Run model -> export active-learning candidates -> simulate -> fill actual values -> append feedback -> rerun model
```

This is the main way to improve real unseen accuracy.

## Feedback CSV Format

`data/feedback_simulations.csv` can contain model prediction columns, but training uses only the actual simulator values.

Required values:

```text
LG, WR, LR, WF, Actual FL, Actual FU, Actual BW
```

Full template format:

```csv
LG,WR,LR,WF,physics_fc,Pred Freq(Lower) GHz,Pred Freq(Upper) GHz,Pred BW GHz,Probability resonant,Expected BW GHz,Combined score,Known in old dataset,Actual FL,Actual FU,Actual BW
```

Example completed row:

```csv
6.5,11.75,3.3,4.7,6.6,5.3658,8.1259,2.8990,0.5211,1.5108,0.6014,False,5.42,7.95,2.53
```

## Compare Baseline vs Feedback

After generating both folders:

```text
research_outputs/
research_outputs_feedback/
```

Run:

```bash
../wide-venv/bin/python compare_feedback_results.py
```

Output:

```text
comparison_outputs/feedback_comparison_report.md
comparison_outputs/tables/metric_before_after_comparison.csv
comparison_outputs/tables/technique_before_after_comparison.csv
comparison_outputs/tables/feedback_prediction_errors.csv
```

Use this to see whether feedback improved accuracy, F1, AUC, and regression error.

## Useful Command Summary

Baseline:

```bash
../wide-venv/bin/python research_pipeline.py --output research_outputs
```

Feedback:

```bash
../wide-venv/bin/python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback
```

Feedback + VLOOKUP + active learning:

```bash
../wide-venv/bin/python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback_vlookup --active-n 20
```

Fast test run:

```bash
../wide-venv/bin/python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback_vlookup --epochs 5 --active-n 20
```

Export 30 active-learning points:

```bash
../wide-venv/bin/python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback_vlookup --active-n 30
```

Compare baseline and feedback:

```bash
../wide-venv/bin/python compare_feedback_results.py
```

## Which File Should You Use For What?

| Goal | File |
|---|---|
| See final metrics | `research_outputs_feedback_vlookup/research_summary.json` |
| See pure ML accuracy | `research_summary.json -> validation_classifier` |
| See VLOOKUP accuracy | `research_summary.json -> vlookup_assisted_validation_classifier` |
| See known lookup table | `tables/known_data_vlookup_table.csv` |
| See next simulations | `tables/active_learning_next_20_simulation_plan.csv` |
| Fill feedback rows | `tables/active_learning_feedback_template_next_20.csv` |
| Add completed simulations | `data/feedback_simulations.csv` |
| Use saved trained model | `models/consolidated_antenna_model.pkl` |
| Compare feedback effect | `comparison_outputs/feedback_comparison_report.md` |

## Practical Notes

- VLOOKUP improves predictions for old/known designs only.
- Feedback improves the actual learned model for unseen designs.
- Active learning is the best way to choose the next simulation points.
- Do not judge the model only by raw accuracy. Also check balanced accuracy, F1, AUC, BW MAE, and BW RMSE.
- If you want better dead/resonant separation, add feedback rows around uncertain probabilities near `0.4` to `0.7`.
- If you want better bandwidth prediction, add feedback rows where predicted `BW` is high but model uncertainty is also high.
