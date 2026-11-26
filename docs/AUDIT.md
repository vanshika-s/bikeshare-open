# Exercise 3: Repository Audit

## Peer Audit

Repository: https://github.com/surai1020/bikeshare-open

### Automated Audit

I used the Claude Agent SDK with the local bikeshare-coder model to audit my classmate's repository. The agent reported the following results:

- Target leakage: None detected
- Temporal validity: Passed
- Baseline comparison: Passed
- Reproducibility: Passed
- Confidence: High
- Findings: None

The agent returned no specific findings or supporting evidence, so I manually reviewed the code to verify its conclusions.

### Manual Verification

I reviewed src/features.py, src/config.py, and src/train.py. The target and its components, casual and registered, are excluded from the feature set. The lagged features and rolling statistics use previous observations rather than the current day's target.

The new feature, cnt_roll_std_7, calculates the standard deviation of bike rental counts over the previous seven days. It correctly applies shift(1) before calculating the rolling standard deviation.

The data is split chronologically, and the models are fitted only on the training data. The StandardScaler is included within the Ridge pipeline, which prevents it from being fitted on validation or test data.

I ran the full test suite and all 18 tests passed. I also ran make all, which successfully reproduced the main modeling pipeline.

### Reproduced Results

| Model | Validation R2 | Validation MAE | Test R2 | Test MAE |
|---|---:|---:|---:|---:|
| Mean baseline | -0.9318 | 2002.2 | -2.1441 | 2847.7 |
| Persistence | 0.4507 | 900.6 | 0.5099 | 840.1 |
| Ridge | 0.7224 | 700.4 | 0.6356 | 840.3 |
| HGB | 0.0977 | 1381.6 | -0.6959 | 2108.4 |

The reproduced results match the main model results discussed in REPORT.md. Ridge has a higher test R2 than persistence, but its test MAE is slightly higher. The report correctly explains that Ridge does not provide a meaningful improvement in average absolute error.

### Findings and Recommendations

1. The target transformation experiments are explained in REPORT.md, but I could not find their implementation in the tracked experiment or training code. I recommend including the code used to generate these results so that others can reproduce the experiments.

2. The pipeline uses same-day weather measurements. This is reasonable if weather information is available when the prediction is made, but the forecasting assumptions should be stated clearly. If predictions are made before the day's weather is observed, weather forecasts would be needed instead.

3. The automated auditor reported high confidence without providing specific evidence. Although its main conclusions were supported by my manual review, it did not identify the missing experiment code.

### Conclusion

I did not find any confirmed target leakage in the main modeling pipeline. The temporal split is correct, the baselines are included, and all existing tests pass. The main pipeline is reproducible, but the target transformation experiments could be better documented by including the implementation.

This audit showed me why it is important to verify an agent's conclusions rather than relying only on its confidence rating.

## Self-Audit

Repository: https://github.com/vanshika-s/bikeshare-open

### Automated Audit

I used the same Claude Agent SDK and local bikeshare-coder model to audit my own repository. The agent reported the following results:

- Target leakage: None detected
- Temporal validity: Passed
- Baseline comparison: Passed
- Reproducibility: Failed
- Confidence: High

### Automated Finding

The agent flagged StandardScaler in src/train.py because it did not have a random_state parameter. It suggested setting random_state=0 to improve reproducibility.

### Manual Verification

I checked the parameters accepted by StandardScaler using Python's inspect module. The output was:

    (*, copy=True, with_mean=True, with_std=True)

StandardScaler does not accept a random_state parameter because it does not use randomness. Adding this parameter would introduce an error rather than improve reproducibility.

I also ran the full modeling pipeline using make all and confirmed that all 17 tests passed. The new weather_severity_roll_3 feature was included in the generated feature set.

To further verify reproducibility, I saved the generated metrics, ran make all again, and compared the two metrics files using diff. The command returned no differences, confirming that both runs produced identical metrics in my environment.

### Decision

I rejected the auditor's recommendation and did not modify StandardScaler. The reported issue was a false positive.

### Reflection

The peer audit and self-audit showed two different limitations of the automated auditor. During the peer audit, the agent missed a reproducibility concern related to the target transformation experiments. During the self-audit, it incorrectly identified a problem with StandardScaler.

In both cases, the agent reported high confidence. This showed me that confidence alone is not enough to determine whether an audit is accurate. Manually reviewing the code and running tests helped me make better decisions about which findings to accept.
