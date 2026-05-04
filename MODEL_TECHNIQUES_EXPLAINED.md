# Model Techniques Explained

This document explains every major technique used in the final research workflow:

```text
research_pipeline.py
```

The project uses a physics-guided machine learning pipeline for antenna bandwidth prediction.

The model has two main jobs:

```text
1. Classification:
   Decide whether an antenna geometry is resonant or dead.

2. Regression:
   If the antenna is resonant, predict FL, FU, and BW.
```

The input geometry is:

```text
LG = ground length
WR = rectangle width
LR = rectangle length
WF = feed width
```

The output is:

```text
FL = lower frequency
FU = upper frequency
BW = bandwidth
```

## Complete Model Flow

```text
Original simulation data
        |
        v
Feature engineering from LG, WR, LR, WF
        |
        v
Classification models predict resonant probability
        |
        v
Stacked classifier combines classifier outputs
        |
        v
Regression models predict FL, FU, BW
        |
        v
Full design grid is ranked
        |
        v
Physics-guided scoring adjusts ranking
        |
        v
Top unseen antenna candidates are recommended
        |
        v
Real simulation feedback is added and model retrains
```

## 1. Feature Engineering

### General Use Case

Feature engineering converts simple raw inputs into more informative variables. It is commonly used when the relationship between inputs and outputs is nonlinear.

For antenna design, small geometry changes can cause nonlinear changes in resonance and bandwidth. Raw `LG`, `WR`, `LR`, and `WF` alone may not expose these patterns clearly.

### What It Does In This Model

The script creates physics-informed features such as:

```text
LG², WR², LR², WF²
LG × WR
WR × WF
LG × WR × WF
WR boundary indicators near WR = 8.5, 10.75, 11.0, 11.25, 12.0
sine/cosine features for resonance-like periodic behavior
geometry ratios such as WR / WF and WR / LG
```

### Contribution To Training

Feature engineering gives the ML algorithms richer inputs. Instead of forcing the model to learn every nonlinear relationship from raw data, some useful relationships are already exposed.

### Contribution To Prediction

For every new candidate antenna point, the same engineered features are created. These features are used by all classifiers and regressors to predict resonance, `FL`, `FU`, and `BW`.

## 2. Extra Trees Classifier

### General Use Case

Extra Trees, or Extremely Randomized Trees, is a tree ensemble method. It is commonly used for tabular datasets with nonlinear relationships and feature interactions.

It works well when:

```text
- the dataset has many engineered features
- relationships are nonlinear
- feature interactions are important
- fast and robust baseline performance is needed
```

### What It Does In This Model

Extra Trees Classifier predicts:

```text
Probability that BW > 0
```

In simple terms, it helps answer:

```text
Will this antenna resonate?
```

### Contribution To Training

It learns many randomized decision trees. Each tree splits the data using different feature thresholds. The final probability is averaged across all trees.

Because antenna behavior can change suddenly near geometry boundaries, tree methods are useful because they can learn threshold-like behavior.

### Contribution To Output Prediction

Extra Trees contributes one probability estimate to the stacked classifier.

Example:

```text
Extra Trees says: probability resonant = 0.62
```

That probability becomes one input to the meta-model.

## 3. Random Forest Classifier

### General Use Case

Random Forest is another tree ensemble method. It is widely used for robust classification on tabular data.

It works well when:

```text
- the data is noisy
- the model should avoid overfitting one tree
- nonlinear interactions are present
- interpretability through feature importance is useful
```

### What It Does In This Model

Random Forest Classifier also predicts:

```text
Probability that the antenna is resonant
```

### Contribution To Training

It trains many decision trees on different bootstrap samples of the training data. This makes it more stable than a single decision tree.

Compared with Extra Trees, Random Forest is usually slightly less random and more conservative.

### Contribution To Output Prediction

Random Forest gives a second independent resonance probability.

Example:

```text
Random Forest says: probability resonant = 0.58
```

This adds another viewpoint to the stacked model.

## 4. Hist Gradient Boosting Classifier

### General Use Case

Histogram Gradient Boosting is a boosting algorithm for structured/tabular data.

It is commonly used when:

```text
- high predictive accuracy is needed
- nonlinear relationships exist
- the model should correct previous mistakes step by step
```

### What It Does In This Model

Hist Gradient Boosting Classifier predicts resonant probability:

```text
Probability that BW > 0
```

### Contribution To Training

Boosting trains trees sequentially. Each new tree focuses on errors made by previous trees.

This is different from Random Forest and Extra Trees, which train many trees independently.

### Contribution To Output Prediction

Hist Gradient Boosting gives a third resonance probability.

Example:

```text
Hist Gradient Boosting says: probability resonant = 0.66
```

This helps the model capture patterns that the bagging-based methods may miss.

## 5. Logistic Regression CV Meta-Learner

### General Use Case

Logistic Regression is commonly used for binary classification. In this project, it is not used directly on antenna geometry. It is used as a meta-learner.

A meta-learner combines predictions from other models.

### What It Does In This Model

The meta-learner receives probability outputs from:

```text
Extra Trees
Random Forest
Hist Gradient Boosting
```

Then it learns how to combine them into one final probability:

```text
Final probability resonant
```

### Contribution To Training

During cross-validation, each base classifier produces out-of-fold probability predictions. These predictions form a small table:

```text
ET probability | RF probability | HGB probability | true label
```

The Logistic Regression CV model learns which base models should be trusted more.

### Contribution To Output Prediction

For a new antenna point:

```text
ET gives probability
RF gives probability
HGB gives probability
Logistic meta-learner combines them
Final resonant probability is produced
```

This is the consolidated classifier.

## 6. Threshold Tuning

### General Use Case

Classification models output probabilities, but final decisions require a threshold.

For example:

```text
if probability >= 0.50, classify as resonant
```

But 0.50 is not always the best threshold.

### What It Does In This Model

The script searches thresholds from `0.10` to `0.90` and chooses a threshold that favors useful resonant antenna discovery.

The model is tuned to avoid missing resonant antennas, because missing a good high-bandwidth candidate is costly.

### Contribution To Training

Threshold tuning selects the operating point of the classifier after the probabilities are learned.

### Contribution To Output Prediction

The final probability is compared with the chosen threshold:

```text
if final probability >= threshold:
    antenna is resonant
else:
    antenna is dead
```

## 7. Extra Trees Regressor

### General Use Case

Extra Trees Regressor predicts continuous numeric values. It is useful for nonlinear regression on tabular data.

### What It Does In This Model

Extra Trees Regressor predicts:

```text
BW
FL
FU
```

There is a separate regressor group for each target.

### Contribution To Training

It learns nonlinear relationships between geometry features and frequency/bandwidth outputs.

It is trained only on resonant rows because dead antennas with `BW = 0` should not dominate frequency prediction.

### Contribution To Output Prediction

For a candidate point, Extra Trees gives one predicted value for each target.

Example:

```text
ET predicts BW = 3.10 GHz
```

This prediction is later combined with Random Forest and HGB regressor predictions.

## 8. Random Forest Regressor

### General Use Case

Random Forest Regressor is used for stable nonlinear numeric prediction.

It is good when:

```text
- data is noisy
- many features interact
- a stable average prediction is preferred
```

### What It Does In This Model

It predicts:

```text
BW
FL
FU
```

### Contribution To Training

It trains many trees on bootstrapped versions of the resonant dataset.

### Contribution To Output Prediction

It gives another estimate of each continuous target.

Example:

```text
RF predicts BW = 2.95 GHz
```

This stabilizes the final regression output.

## 9. Hist Gradient Boosting Regressor

### General Use Case

Hist Gradient Boosting Regressor is used for high-quality nonlinear regression.

It is useful when:

```text
- residual errors need to be corrected
- input-output relationships are complex
- tabular prediction accuracy matters
```

### What It Does In This Model

It predicts:

```text
BW
FL
FU
```

### Contribution To Training

It learns sequential trees where each step tries to correct the previous prediction errors.

### Contribution To Output Prediction

It contributes a third estimate of each output.

Example:

```text
HGB predicts BW = 3.25 GHz
```

## 10. Weighted Regression Ensemble

### General Use Case

A weighted ensemble combines multiple model predictions into one final prediction. This is often more stable than using one model alone.

### What It Does In This Model

For each target, the final prediction is:

```text
Final prediction = 0.40 × Extra Trees
                 + 0.35 × Random Forest
                 + 0.25 × Hist Gradient Boosting
```

This is done separately for:

```text
BW
FL
FU
```

### Contribution To Training

The individual regressors are trained separately. The fixed weights define how their outputs are combined.

### Contribution To Output Prediction

For a candidate point:

```text
ET predicts BW
RF predicts BW
HGB predicts BW
Weighted average gives final BW
```

Same for `FL` and `FU`.

## 11. MLP Epoch Monitor

### General Use Case

MLP means Multi-Layer Perceptron, a basic neural network. Neural networks are often used to learn nonlinear relationships.

In this project, the MLP is not the final main model. It is used as a learning monitor.

### What It Does In This Model

It trains over multiple epochs and records:

```text
training accuracy
validation accuracy
training loss
validation loss
```

### Contribution To Training

It helps create the learning-over-epochs plot required for the paper.

### Contribution To Output Prediction

The MLP monitor does not produce the final antenna recommendations. Its main purpose is diagnostic plotting.

Final recommendations come from:

```text
stacked classifier + weighted regression ensemble + physics-guided ranking
```

## 12. Quantile Transformer

### General Use Case

Quantile Transformer converts feature distributions into a more regular distribution, usually close to Gaussian.

It is useful for neural networks because neural networks train better when features are scaled and well distributed.

### What It Does In This Model

It is used before the MLP epoch monitor.

### Contribution To Training

It stabilizes neural-network training.

### Contribution To Output Prediction

It only affects the MLP learning-curve monitor, not the final consolidated model.

## 13. Physics-Guided Scoring

### General Use Case

Physics-guided scoring uses approximate domain knowledge to make ML recommendations more physically meaningful.

It is useful when pure ML may over-rank points that look statistically good but may be physically less plausible.

### What It Does In This Model

The script estimates effective permittivity from known resonant data:

```text
eps_eff ≈ (150 / (patch_length × center_frequency))²
```

Then it computes guided wavelength:

```text
lambda_g = 300 / (frequency × sqrt(eps_eff))
```

Then it compares geometry ratios:

```text
WR / lambda_g
LG / lambda_g
WF / lambda_g
```

Candidate points score higher if their wavelength ratios are close to high-bandwidth designs already observed in the simulation data.

### Contribution To Training

Physics-guided scoring is not used to directly train the classifier or regressors. It is used after ML prediction to rank candidate designs.

### Contribution To Output Prediction

The ML model predicts:

```text
probability resonant
FL
FU
BW
```

Physics-guided scoring then helps decide which predicted high-BW points are more physically plausible.

Final ranking uses both:

```text
ML prediction strength + physics-guided score
```

## 14. Feedback Learning

### General Use Case

Feedback learning means taking real-world or simulator-verified results and adding them back into the training data.

This is useful when the model makes a prediction, then a real simulation shows the actual result.

### What It Does In This Model

You add feedback rows to:

```text
data/feedback_simulations.csv
```

Then run:

```bash
python research_pipeline.py --feedback data/feedback_simulations.csv --output research_outputs_feedback
```

The model retrains with:

```text
original data + feedback data
```

### Contribution To Training

Feedback corrects the model in regions where it was wrong.

Example:

```text
Model predicted BW = 3.2365 GHz
Actual BW = 1.41 GHz
```

After feedback, the model learns not to overestimate that geometry region as much.

### Contribution To Output Prediction

Feedback changes future candidate rankings and output predictions. Points similar to the feedback data are adjusted based on actual simulator behavior.

## 15. Technique Contribution Summary

| Technique | General Use | Role In This Project | Contribution To Final Output |
|---|---|---|---|
| Feature engineering | Convert raw inputs into informative variables | Encodes nonlinear and physics-inspired geometry relationships | Improves all classifiers and regressors |
| Extra Trees Classifier | Robust nonlinear classification | Predicts resonant probability | Base probability for stacked classifier |
| Random Forest Classifier | Stable tabular classification | Predicts resonant probability | Base probability for stacked classifier |
| Hist Gradient Boosting Classifier | Sequential error-correcting classifier | Predicts resonant probability | Base probability for stacked classifier |
| Logistic Regression CV | Probability combiner/meta-model | Combines classifier outputs | Produces final resonant probability |
| Threshold tuning | Choose classification decision boundary | Favors useful resonant candidate discovery | Converts probability into resonant/dead label |
| Extra Trees Regressor | Nonlinear numeric prediction | Predicts FL/FU/BW | One part of weighted regression output |
| Random Forest Regressor | Stable numeric prediction | Predicts FL/FU/BW | One part of weighted regression output |
| Hist Gradient Boosting Regressor | Corrects regression residuals | Predicts FL/FU/BW | One part of weighted regression output |
| Weighted regressor ensemble | Combine numeric predictions | Blends ET/RF/HGB regressors | Final FL/FU/BW values |
| MLP monitor | Neural learning diagnostic | Produces epoch accuracy/loss plots | Diagnostic only, not final candidate ranking |
| Quantile Transformer | Feature scaling for neural network | Stabilizes MLP monitor | Diagnostic only |
| Physics-guided scoring | Domain-informed ranking | Scores candidates by wavelength-ratio plausibility | Adjusts final unseen candidate ranking |
| Feedback learning | Improve from real simulations | Adds actual tested rows into training data | Updates model and future rankings |

## Final Prediction Logic

For a new antenna candidate:

```text
1. Build features from LG, WR, LR, WF.
2. ET classifier predicts resonance probability.
3. RF classifier predicts resonance probability.
4. HGB classifier predicts resonance probability.
5. Logistic meta-learner combines those probabilities.
6. Threshold decides resonant/dead.
7. If resonant, ET/RF/HGB regressors predict FL, FU, BW.
8. Weighted average gives final FL, FU, BW.
9. Candidate is scored using expected BW and physics-guided score.
10. Top candidates are exported for real simulation.
```

## Best Short Description For Paper

You can describe the final model like this:

> The proposed framework uses a stacked ensemble classifier to identify resonant antenna geometries and a weighted ensemble regressor to estimate lower frequency, upper frequency, and bandwidth. The candidate search is further refined using physics-guided wavelength-ratio scoring, and real EM simulation feedback is incorporated into retraining to improve subsequent predictions.

