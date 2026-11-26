# Bikeshare Demand Forecasting

## 1. Results

The goal of this project is to predict daily bike rental
counts using weather, calendar information, and historical
rental demand.

The original dataset contains 728 rows. After calendar
reindexing and feature construction, 714 observations
remain, with 348 training, 182 validation, and 184 test
observations. The final model uses 15 features.

The data is split chronologically, with 2011 used for
training, the first half of 2012 for validation, and the
second half of 2012 for testing.

| Model | Validation R2 | Validation MAE | Test R2 | Test MAE |
|---|---:|---:|---:|---:|
| Mean baseline | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Ridge | 0.6639 | 809.8 | 0.6022 | 920.3 |
| HGB | 0.2160 | 1299.0 | -0.4975 | 1975.6 |

I added weather_severity_roll_3, which calculates the
average weather severity over the previous three calendar
days. Ridge validation R2 increased from 0.6475 to 0.6639,
while HGB validation R2 decreased from 0.2185 to 0.2160.
I retained the feature based on the Ridge validation
improvement.

## 2. Headline Finding

A model that uses casual and registered rental counts
can achieve an R2 of 1.000 because these two columns sum
exactly to the target. This is target leakage rather than
successful forecasting.

After excluding forbidden features and using a temporal
split, Ridge achieves a test R2 of 0.6022. Although this
is higher than the persistence baseline, Ridge has a
higher test MAE.

The negative R2 of the mean baseline also shows how
much rental demand changed between the training and test
periods. Gradient boosting struggles with this change
because tree-based models have difficulty extrapolating
beyond their training distribution.

These results show why a high model score alone is not
enough to establish that a forecasting approach is useful.

## 3. Exercise 4: Target Transformation

### Problem

The original gradient boosting model struggles to predict
bike rental counts in 2012 because demand increased compared
to 2011. The training target ranges from 431 to 6043 rentals,
while the validation target ranges from 1027 to 8362 rentals.
Out of 182 validation days, 71 days (39%) have rental counts
above the maximum training value.

### Method

I compared the original gradient boosting model with two
target transformations. The first uses log1p(cnt) and
converts predictions back using expm1. The second predicts
log(cnt / cnt_lag_1), which represents the log relative
change in rentals compared to the previous day.

The relative growth predictions are converted back to
rental counts by applying the exponential function and
multiplying by the previous day's actual rental count.

All three models use the same 15 features and model
settings. They were fitted on the training set and evaluated
on the validation set. All metrics were calculated on the
original rental count scale. The test set was not used to
evaluate or select the transformations.

### Results

| Approach | Validation R2 | Validation MAE | Mean Error |
|---|---:|---:|---:|
| Original HGB | 0.2160 | 1299.0 | -1196.2 |
| Log transformation | 0.1680 | 1341.6 | -1299.9 |
| Relative growth | 0.5877 | 866.7 | -575.3 |

The log transformation performed slightly worse than
the original model. Both underestimated rental demand
on average.

The relative growth transformation performed better,
reducing MAE by approximately 33.3% compared to the
original gradient boosting model. Predicting changes
relative to the previous day's demand helps the model
adapt to higher rental counts in 2012.

However, the relative growth model still underestimates
demand on average, so it does not completely solve the
distribution shift problem. This approach also assumes
that the previous day's actual rental count is available
when making a prediction.

The experiment can be reproduced using:

    python -m src.target_transform

The implementation is saved in src/target_transform.py
and the results are saved in reports/target_transform.json.

## 4. Exercise 5: Ridge vs Persistence

| Model | Validation R2 | Validation MAE | Test R2 | Test MAE |
|---|---:|---:|---:|---:|
| Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Ridge | 0.6639 | 809.8 | 0.6022 | 920.3 |

Ridge performs better than persistence on both validation
metrics. However, the test results show a different pattern.
Ridge improves test R2 by 0.0923, but its MAE increases by
80.2 rentals, which is approximately a 9.55% increase.

R2 and MAE measure prediction errors differently. R2 is
based on squared errors, which give more weight to larger
errors. MAE measures the average absolute difference
between predicted and actual rental counts.

Ridge can have a higher R2 while still having a worse MAE
because reducing squared errors does not necessarily
reduce average absolute errors. Further analysis of
individual predictions would be needed to understand
which days contributed most to this difference.

If the main objective is to minimize average absolute
prediction error, I would recommend the persistence
baseline. Its lower test MAE shows why comparing a model
against a simple baseline is important before deciding
whether additional complexity is worthwhile.

## 5. Attribution

### Agent Contributions

I used Claude Code with the local bikeshare-coder model,
built from qwen3.5:9b using Ollama. I also used ChatGPT for troubleshooting, verification, and report editing.

Claude Code assisted with the feature verification,
seeded fault experiments, target transformation experiment,
and interpretation of model results. I used the Claude
Agent SDK to audit my repository and a classmate's
repository.

### My Decisions and Verification

I chose to investigate historical weather severity and
retained the new feature based on its validation results.
I verified the feature calculation and its behavior
when calendar dates are missing.

During the seeded fault experiments, I checked which
errors the existing guardrails detected. I added two
additional tests to address gaps in the original test
coverage.

I also reviewed the automated audit findings manually.
The self-audit incorrectly suggested adding a random_state
parameter to StandardScaler. I verified that StandardScaler
does not accept this parameter and confirmed that repeated
pipeline runs produced identical metrics in my environment.

During the target transformation experiment, I corrected
the implementation with additional guidance to remove
the dependency on validation targets during prediction,
reuse the existing temporal split function, and save all
three sets of results.

A separate Claude Code review found no confirmed issues
with the corrected transformation experiment. Some
verification commands failed, so I also relied on the
successful experiment run and manual checks.

I verified the Ridge and persistence comparison using
the generated metrics rather than accepting the agent's
calculations directly.

### Custom Skill

I created a custom skill called compare-models in
.claude/skills/compare-models/SKILL.md.

The skill uses Python to calculate performance differences
from existing metrics without retraining models.

| Comparison | Split | R2 Difference | MAE Change |
|---|---|---:|---:|
| Ridge vs Mean | Validation | +1.5957 | -59.55% |
| Ridge vs Mean | Test | +2.7463 | -67.68% |
| Ridge vs Persistence | Validation | +0.2132 | -10.08% |
| Ridge vs Persistence | Test | +0.0923 | +9.55% |

I verified these calculations against the saved metrics.
Using Python made the comparison more reliable, although
interpreting the results still required human review.

### Supporting Documentation

The methodological decisions are recorded in
docs/DECISIONS.md. The seeded fault experiments are
documented in docs/SEEDED_FAULTS.md, and the peer audit
and self-audit are documented in docs/AUDIT.md.

The additional leakage reviewer attempt was stopped
before all five checks were completed.

The latest full test run passed all 20 tests. This project
helped me understand that an agent producing working code
does not automatically mean that its results are valid.
Tests and independent verification are important for
determining whether the results can be trusted.
