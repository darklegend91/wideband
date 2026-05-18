# Wideband Antenna Research Pipeline

This repository contains a single-file workflow for training and evaluating a consolidated ML model for wideband antenna bandwidth prediction.

The main script is:

```bash
research_pipeline.py
```

For a beginner-friendly explanation of every folder and the full run workflow, read:

```text
folder_running_guide.md
```

For the current practical command guide covering feedback, VLOOKUP, and active learning, read:

```text
USAGE_GUIDE.md
```

For a plain-language explanation of each ML technique and its role in the final model, read:

```text
MODEL_TECHNIQUES_EXPLAINED.md
```

It uses the 4 antenna design inputs `LG`, `WR`, `LR`, and `WF` to predict:

- Resonant vs dead antenna behavior
- `FL`, lower operating frequency
- `FU`, upper operating frequency
- `BW`, bandwidth

## Setup

Create and activate a Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

From the repository root:

```bash
python research_pipeline.py
```

If your dataset is somewhere else:

```bash
python research_pipeline.py --data data/UBW-data-final.xlsx
```

Useful options:

```bash
python research_pipeline.py --epochs 20 --top-n 25 --output research_outputs
```

## Feeding Back Real Simulation Results

When you test a model-recommended point in HFSS/CST or another EM simulator, add the actual result to:

```text
data/feedback_simulations.csv
```

The script automatically loads this file when it exists. The feedback file can contain the model prediction columns plus the actual simulator outputs:

```text
LG, WR, LR, WF, physics_fc,
Pred Freq(Lower) GHz, Pred Freq(Upper) GHz, Pred BW GHz,
Probability resonant, Expected BW GHz, Combined score,
Known in old dataset, Actual FL, Actual FU, Actual BW
```

Only the actual simulator values are used as training targets:

```text
Actual FL -> FL
Actual FU -> FU
Actual BW -> BW
```

Retrain after adding feedback:

```bash
python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback
```

For your current feedback point, the model prediction was:

```text
LG=5.5, WR=12.25, LR=3.0, WF=3.6
Predicted: FL=5.2731 GHz, FU=8.3073 GHz, BW=3.2365 GHz
Actual:    FL=5.29 GHz,   FU=6.70 GHz,   BW=1.41 GHz
```

This teaches the model that this region was overestimating upper frequency and bandwidth.

## Outputs

All generated results are written to:

```text
research_outputs/
```

Important files:

```text
research_outputs/research_summary.json
research_outputs/models/consolidated_antenna_model.pkl
research_outputs/tables/known_data_vlookup_table.csv
research_outputs/tables/validation_vlookup_assisted_predictions.csv
research_outputs/tables/active_learning_next_20_simulation_plan.csv
research_outputs/tables/active_learning_feedback_template_next_20.csv
research_outputs/tables/technique_comparison_metrics.csv
research_outputs/tables/validation_regression_predictions.csv
research_outputs/tables/top_100_unseen_grid_predictions.csv
research_outputs/tables/physics_guided_top_25_unseen_predictions.csv
```

## Known-Data VLOOKUP Layer

The pipeline now builds a VLOOKUP-style table from every simulated antenna geometry:

```text
LG, WR, LR, WF -> FL, FU, BW, resonant/dead label
```

When a prediction request exactly matches an older simulated design, the saved model artifact returns the known simulator result from the lookup table. When the design is not in old data, it falls back to the ML classifier and regressors.

This gives exact known-data metrics:

```text
vlookup_known_data_classifier accuracy = 100%
vlookup_known_data_regression MAE = 0
```

Keep this separate from the pure ML validation score. The VLOOKUP score measures memory over older simulated rows; the pure ML validation score measures how well the learned model generalizes without using the lookup answer.

## Active Learning Feedback Loop

Each run exports the next recommended simulation batch:

```text
research_outputs/tables/active_learning_next_20_simulation_plan.csv
research_outputs/tables/active_learning_feedback_template_next_20.csv
```

The selection combines:

- high predicted bandwidth
- resonant/dead boundary uncertainty
- disagreement between regression models
- physics-guided score

Simulate those antenna geometries in HFSS/CST, fill only these columns in the feedback template:

```text
Actual FL, Actual FU, Actual BW
```

Append those completed rows to:

```text
data/feedback_simulations.csv
```

Then rerun the pipeline. To export a different batch size:

```bash
python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback_vlookup --active-n 30
```

Important plots for the paper:

```text
research_outputs/plots/epoch_accuracy_loss.png
research_outputs/plots/epoch_monitor_confusion_matrix.png
research_outputs/plots/consolidated_confusion_matrix.png
research_outputs/plots/validation_probability_distribution.png
research_outputs/plots/technique_comparison.png
research_outputs/plots/technique_accuracy_loss_progression.png
research_outputs/plots/regression_actual_vs_predicted.png
research_outputs/plots/bandwidth_residuals.png
research_outputs/plots/feature_importance_top20.png
research_outputs/plots/known_vs_predicted_best_bandwidth.png
research_outputs/plots/top_physics_guided_candidates.png
```

## What The Script Does

1. Loads `UBW-data-final.xlsx`.
2. Cleans duplicate antenna configurations by keeping the row with highest `BW`.
3. Builds physics-informed features from `LG`, `WR`, `LR`, and `WF`.
4. Trains an epoch-based MLP monitor for accuracy/loss learning curves.
5. Trains base classifiers: Extra Trees, Random Forest, and Hist Gradient Boosting.
6. Combines them with a logistic-regression stacking model.
7. Compares each individual technique with the consolidated stacked model.
8. Trains ensemble regressors for `BW`, `FL`, and `FU`.
9. Ranks unseen antenna parameter combinations.
10. Adds a physics-guided score using wavelength-ratio heuristics.

## Dataset Format

The Excel file must contain these columns in order:

```text
SrNo, LG, WR, LR, WF, FL, FU, BW
```

`BW > 0` is treated as resonant. `BW = 0` is treated as dead/non-resonant.

## Recommended Paper Figures

Use these figures directly in your results section:

- `consolidated_confusion_matrix.png` for final classification quality
- `epoch_accuracy_loss.png` for learning behavior over epochs
- `technique_comparison.png` to show why the consolidated model is stronger than single techniques
- `regression_actual_vs_predicted.png` for `BW`, `FL`, and `FU` regression fit
- `known_vs_predicted_best_bandwidth.png` to compare known best designs with predicted unseen candidates
- `top_physics_guided_candidates.png` to support final design recommendations


# Some Doubts


• The “points” are antenna parameter combinations from your design grid.

  Each point means one possible antenna geometry:

  LG = ground length
  WR = rectangle width
  LR = rectangle length
  WF = feed width

  Example:

  LG=5.5, WR=12.25, LR=3.0, WF=3.5

  That is one proposed antenna design.

  Your model is doing this:

  1. It reads your simulated dataset from data/UBW-data-final.xlsx.
  2. It learns from already simulated antenna designs:
      - inputs: LG, WR, LR, WF
      - outputs: FL, FU, BW
      - classification label: resonant if BW > 0, dead if BW = 0
  3. It creates many physics-inspired features from those four inputs, for example:
      - squared/cubic terms like LG², WR²
      - interactions like LG × WR, WR × WF
      - WR boundary features near important values like WR=11
      - sine/cosine features to capture resonance-like patterns
  4. It trains classification models to answer:

     Will this antenna resonate or not?

  5. It trains regression models to answer:

     If it resonates, what are FL, FU, and BW?

  6. It generates all possible candidate points from this grid:

  LG: 10 values
  WR: 18 values
  LR: 2 values
  WF: 31 values

  So total possible designs:

  10 × 18 × 2 × 31 = 11,160 antenna points

  7. It removes points already present in your Excel dataset.
  8. For the remaining unseen points, it predicts:
      - probability of resonance
      - predicted FL
      - predicted FU
      - predicted BW
      - expected bandwidth score
      - physics-guided score

  So the top recommended points are not random. They are unseen antenna geometries selected from the full design grid because your model predicts they may give high bandwidth and reasonable operating frequency behavior.

  In short:

  Your existing HFSS/CST simulation data teaches the model antenna behavior.
  The model searches the untested design grid.
  It recommends new geometry points that are likely to produce strong bandwidth.

  Important paper wording:

  > The proposed model uses prior simulated antenna responses to learn the nonlinear relationship between geometric parameters and antenna bandwidth. It then evaluates the untested design space and ranks candidate geometries using a
  > consolidated ensemble prediction and physics-guided wavelength-ratio scoring.


› so my points are calcluated by model not by physics formulas and how physics is playing its role here ?


• Yes: the final FL, FU, BW, and resonance probability are calculated by the ML model, not directly by physics formulas.

  Physics is playing a supporting/ranking role, not replacing the model.

  The flow is:

  1. Generate possible antenna geometries from your design grid
     LG, WR, LR, WF

  2. Remove geometries already present in your dataset

  3. ML model predicts for each unseen geometry:
     probability of resonance
     predicted FL
     predicted FU
     predicted BW

  4. Physics-guided scoring checks whether each geometry is close to patterns seen in good antennas:
     WR / guided wavelength
     LG / guided wavelength
     WF / guided wavelength

  5. Final ranking combines:
     ML predicted bandwidth
     ML resonance probability
     model confidence
     physics-guided score

  So the model says:

  > “This point may give high bandwidth.”

  The physics score says:

  > “This point also looks physically reasonable compared with high-bandwidth resonant antennas already observed.”

  Physics is used like a filter or guide to prefer candidates that make electromagnetic sense.

  In research_pipeline.py, the physics part estimates an effective permittivity from your known resonant data:

  eps_eff ≈ (150 / (patch_length × center_frequency))²

  Then it estimates guided wavelength:

  lambda_g = 300 / (frequency × sqrt(eps_eff))

  Then it compares each candidate geometry using ratios:

  WR / lambda_g
  LG / lambda_g
  WF / lambda_g

  The best known high-bandwidth antennas have typical wavelength ratios. New candidate points get higher physics scores if their ratios are close to those high-bandwidth patterns.

  So for your paper, describe it like this:

  > The proposed approach is not purely formula-based. The primary prediction of resonance, lower frequency, upper frequency, and bandwidth is performed by the trained ensemble ML model. Physics is incorporated as a guidance layer through
  > effective-permittivity and guided-wavelength ratio features, which bias the ranking of unseen candidate designs toward geometries consistent with high-bandwidth resonant behavior observed in the simulation data.

  Important distinction:

  ML model = predicts performance
  Physics formulas = guide candidate ranking and improve plausibility

  This is why it is fair to call it a physics-guided machine learning model, not a pure physics model.
