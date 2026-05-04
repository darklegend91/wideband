# Physics-Guided Unseen Frequency Prediction Files

This folder collects the files used by the physics-guided unseen-frequency prediction workflow. Originals were copied, not moved.

## data

- `UBW-data-final.xlsx` - Original antenna simulation dataset used by the workflow.

## workflow

- `physics_guided_unseen_frequency_predictions.ipynb` - Main physics-guided unseen candidate generation and prediction notebook.
- `high_bw_resonant_model.ipynb` - Upstream notebook that trained the high-bandwidth resonant model used by the physics-guided workflow.
- `train_base_model_from_scratch.ipynb` - Clean scratch-training notebook that reads the dataset, trains the base classifier/regressors, saves fresh model artifacts, and generates maximum-output candidate rankings.

## code

- `high_bw_resonant_model.py` - Python script export of the upstream model-training notebook.
- `physics_guided_unseen_frequency_predictions.py` - Python script export of the physics-guided prediction notebook.

## model

- `high_bw_resonant_model.pkl` - Trained classifier/regressor artifact loaded by the physics-guided notebook.
- `high_bw_model_summary.json` - Model summary and validation metrics.
- `high_bw_classifier_metrics.csv` - Classifier validation metrics.
- `resonant_threshold_search.csv` - Threshold search table used for resonant/dead classification.

## predictions

- `physics_calibration_summary.json` - Physics calibration values estimated from known resonant rows.
- `physics_guided_summary.json` - Final physics-guided run summary and best recommendation.
- `physics_guided_top_100_unseen_predictions.csv` - Top 100 unseen predictions.
- `physics_guided_top_25_unseen_predictions.csv` - Top 25 unseen predictions.
- `physics_guided_top_20_unseen_predictions.csv` - Top 20 unseen predictions.
- `top_5_max_bw_predictions.csv` - Top 5 maximum-bandwidth predictions.
- `top_10_max_bw_predictions.csv` - Top 10 maximum-bandwidth predictions.
- `top_25_max_bw_predictions.csv` - Top 25 maximum-bandwidth predictions.
- `top_5_full_sheet_parameters.csv` - Top 5 predictions expanded to full antenna sheet parameters.
- `top_10_full_sheet_parameters.csv` - Top 10 predictions expanded to full antenna sheet parameters.
- `top_25_full_sheet_parameters.csv` - Top 25 predictions expanded to full antenna sheet parameters.
- `top_50_high_bw_all_grid.csv` - Upstream high-bandwidth predictions over the full grid.
- `top_50_high_bw_untested.csv` - Upstream high-bandwidth predictions excluding known dataset rows.

## docs

- `readme.md` - Project-level explanation of the broader modeling pipeline.
