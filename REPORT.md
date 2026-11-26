
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

