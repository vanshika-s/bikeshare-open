# Decisions

One line per methodological decision: what you chose, why, and what it changed.
Agents append here via the `/new-feature` skill; you append here by hand.

- Exclude `casual`, `registered`: they sum to `cnt` exactly. Leaky model R^2 1.000 -> honest ridge 0.583.
- Exclude `cnt_2d_bfr`: a shipped 2-day lag we cannot test the alignment of. Build our own lags instead.
- Exclude `instant`, `yr`, `days_since_2011`: time indexes let a tree predict a flat constant past 2011 and a linear model extrapolate a trend we have no grounds for.
- Split 2011 / H1 2012 / H2 2012 by date. Use validation results for model development and test results for final evaluation. Repeat test evaluation only for verification, not tuning.
- Report mean and persistence baselines with every score. Persistence (test R^2 0.510, MAE 840) has a test R^2 that is 0.0923 lower than ridge and beats it on MAE; that caveat goes in the report.

### Exercise 1: Adding a New Feature

**Feature chosen:** `weather_severity_roll_3`

I chose to add a feature that looks at the average weather severity over the previous three days. The idea was to see whether recent weather patterns could help the model predict bike rentals.

The feature uses the weather severity values from the three days before the prediction date. I used `shift(1)` so the calculation would not include the current day's weather. I also used the existing calendar reindexing step to make sure missing dates would not cause the model to use the wrong days.

**Testing process**

I wrote three new tests before adding the feature. The first test checked whether the feature existed. The second checked whether the calculation used the correct dates and excluded the current day. The last test checked how the feature handled a missing calendar date.

Initially, all three tests failed because the feature had not been added yet. After implementing it, two tests passed, but the missing-date test still failed. I realized that the pipeline also uses a seven-day rolling average of bike rentals, which meant the missing date affected more rows than I originally expected. I updated the test to account for this.

**Final results**

After making these changes, all 10 tests in `test_leakage.py` passed. I also ran the full test suite, and all 17 tests passed.

This helped me understand why writing tests before implementing a feature is useful. It gave me a way to check whether the feature worked as intended and helped me catch a mistake in my test.

### Additional Methodological Decisions

- Keep the three-day weather severity feature. Using the same training and validation rows, Ridge validation R2 increased from 0.6475 to 0.6639 (+0.0164), while HGB decreased from 0.2185 to 0.2160 (-0.0025). The feature was retained based on the Ridge validation improvement.

- Add a raw schema contract test so unexpected columns cannot appear in the dataset without being detected by the test suite.

- Use relative growth as an alternative target for gradient boosting. Validation MAE decreased from 1299.0 to 866.7 compared with the original target.

- Keep persistence as the preferred model when minimizing MAE. Ridge improves test R2 by 0.0923 but increases test MAE by 9.55%.

- Verify automated audit findings before changing the implementation. StandardScaler is deterministic and does not accept a random_state parameter.

- Use Python for calculations in the compare-models skill to improve the reliability of model comparisons.

- Keep the original temporal split and avoid tuning models using test results.

### Exercise 1: TDD Commit History

The original feature and tests were committed together
in f656469 after completing the testing process.

I reconstructed the red and green stages on a separate
branch called exercise-1-tdd-reconstruction using the
original committed tests and implementation.

- Red commit (1afc03e): 3 failed, 7 passed.
- Green commit (1971e2d): 10 passed.

These commits demonstrate the testing process retrospectively
without changing the original main branch history.

Branch:
https://github.com/vanshika-s/bikeshare-open/tree/exercise-1-tdd-reconstruction
