---
name: compare-models
description: Compare model performance against baselines using verified calculations and existing evaluation metrics.
allowed-tools: Read, Grep, Glob, Bash(python -c *)
---

# Model comparison: $ARGUMENTS

Follow these steps in order.

1. Read CLAUDE.md, src/train.py, src/features.py,
   src/config.py, and reports/metrics.json.

2. Verify the temporal split and feature selection.
   If the metrics file is missing or incomplete, stop
   and report the problem. Do not retrain models.

3. Use Python to read reports/metrics.json and calculate
   the following for each requested model against both
   the mean and persistence baselines:

   R2 difference = model R2 - baseline R2

   MAE percentage change =
   100 * (model MAE - baseline MAE) / baseline MAE

   Calculate these separately for validation and test.
   Do not perform arithmetic mentally.
   Round R2 differences to four decimal places and
   MAE percentage changes to two decimal places.

4. Present only the requested models in the results.
   Copy numerical results directly from the Python output.
   A positive R2 difference indicates improvement.
   A positive MAE percentage change indicates worse performance.

5. Explain that R2 is based on squared errors, while MAE
   measures average absolute prediction error.

6. Do not infer overfitting, holiday-specific errors,
   individual residuals, causal effects, or feature
   importance from aggregate metrics alone.

7. Recommend the model with lower MAE when minimizing
   average absolute prediction error. Recommend the model
   with higher R2 when R2 is the stated objective.
   If the metrics disagree, explain the tradeoff.

8. Before responding, verify every reported number against
   the Python output. Keep the response concise.

9. Do not modify files, retrain models, tune parameters,
   or make commits.
