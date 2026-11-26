
## Exercise 4: Target Transformation

### Problem

The original gradient boosting model struggles to predict bike rental counts in 2012 because demand increased compared to 2011. The training target ranges from 431 to 6,043 rentals, while the validation target ranges from 1,027 to 8,362 rentals. Out of 182 validation days, 71 days (39%) have rental counts above the maximum training value.

Tree-based models struggle to extrapolate beyond the patterns observed during training. I wanted to see whether transforming the target variable could help the model adapt to this increase in demand.

### Method

I compared the original gradient boosting model with two target transformations. The first transformation uses log1p(cnt) and converts the predictions back using expm1. The second transformation predicts log(cnt / cnt_lag_1), which represents the relative change in rental counts compared to the previous day.

All three models use the same 15 features and model settings. The models were fitted on the training set and evaluated on the validation set. All metrics were calculated on the original rental count scale. The test set was not used to evaluate or select the transformations.

### Results

| Approach | Validation R2 | Validation MAE | Mean Error |
|---|---:|---:|---:|
| Original HGB | 0.2160 | 1299.0 | -1196.2 |
| Log transformation | 0.1680 | 1341.6 | -1299.9 |
| Relative growth | 0.5877 | 866.7 | -575.3 |

The log transformation performed slightly worse than the original model. Both models consistently underestimated rental demand on average.

The relative growth transformation performed better, reducing MAE by approximately 33.3% compared to the original gradient boosting model. Instead of predicting the total number of rentals directly, the model learns how demand changes relative to the previous day. This helps it adapt to higher rental counts in 2012.

However, the relative growth model still underestimates demand on average, so the transformation does not completely solve the distribution shift problem.

### Agent Interaction

I used Claude Code with the local bikeshare-coder model to implement the experiment. The agent initially produced the expected model results, but there were several issues with its implementation. It incorrectly reported the log-transformed target ranges, used validation target values when determining which predictions to make, and had problems saving the complete results.

I manually corrected the script to reuse the existing temporal split function, remove the dependency on validation targets during prediction, and save all three approaches in the results file. After rerunning the experiment, the model results remained unchanged and all 17 existing tests passed.

The experiment can be reproduced by running:

    python -m src.target_transform

The implementation is saved in src/target_transform.py, and the results are generated in reports/target_transform.json.


### Independent Review

After correcting the implementation, I started a new Claude Code session to review the experiment without making further changes. The agent checked the feature selection, temporal split, target transformations, evaluation metrics, and reported results.

The agent found no confirmed issues and reported that all 17 tests passed. Some shell commands failed during the review, so I also relied on my earlier manual verification and successful experiment run.

The relative growth approach assumes that the previous day's actual rental count is available when making a prediction.

No further changes to the implementation were needed after the review.

## Exercise 5: Ridge vs. Persistence

### Results

| Model | Validation R2 | Validation MAE | Test R2 | Test MAE |
|---|---:|---:|---:|---:|
| Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Ridge | 0.6639 | 809.8 | 0.6022 | 920.3 |

Ridge performs better than persistence on both validation metrics. However, the test results show a different pattern. Ridge improves test R2 by 0.0923, but its MAE increases by 80.2 rentals, which is approximately a 9.5% increase.

### Interpretation

R2 and MAE measure prediction errors differently. R2 is based on squared errors, which give more weight to larger errors. MAE measures the average absolute difference between predicted and actual rental counts.

Ridge can have a higher R2 while still having a worse MAE because reducing squared errors does not necessarily reduce average absolute errors. Further analysis of individual predictions would be needed to understand which days contributed most to this difference.

### Conclusion

Although Ridge performs better on the validation set, its improvement does not carry over to both test metrics. If the main objective is to minimize average absolute prediction error, I would recommend the persistence baseline.

Persistence achieves a lower test MAE by simply predicting the previous day's rental count. These results show why comparing a model against a simple baseline is important before deciding whether the additional complexity is worthwhile.

### Agent Interaction

I asked Claude Code to interpret the results and explain the difference between R2 and MAE. The agent correctly identified the overall performance tradeoff and recommended persistence when minimizing MAE.

I verified the calculations using reports/metrics.json and corrected the absolute difference in test MAE. I also revised the explanation of extrapolation because Ridge, unlike tree-based models, can extrapolate beyond the training target range.

No additional model tuning was performed for this comparison.

## Exercise 6: Custom Skill

I created a custom skill called compare-models in
.claude/skills/compare-models/SKILL.md.

The purpose of this skill is to make model comparisons more
consistent and reliable. It instructs the agent to read the
existing metrics and use Python to calculate differences
in model performance instead of calculating them manually.

### Testing the Skill

I tested the skill using the following command in Claude Code:

/compare-models Compare Ridge against the mean and persistence baselines using the existing metrics.

| Comparison | Split | R2 Difference | MAE Change |
|---|---|---:|---:|
| Ridge vs Mean | Validation | +1.5957 | -59.55% |
| Ridge vs Mean | Test | +2.7463 | -67.68% |
| Ridge vs Persistence | Validation | +0.2132 | -10.08% |
| Ridge vs Persistence | Test | +0.0923 | +9.55% |

The final version of the skill produced correct calculations
and recommended persistence when minimizing MAE. I verified
the results against the existing metrics and checked the
Git working tree for unexpected changes.

### Reflection

Using Python for calculations made the skill more reliable.
The skill also provides a consistent structure for comparing
models without retraining them or changing the existing code.
However, the interpretation of results still requires human
review, since aggregate metrics alone cannot explain the
performance of individual features.
