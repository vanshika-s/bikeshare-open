# Seeded Faults

## Fault 1: Rolling Average Includes Today's Target

**File:** `src/features.py`

**Change made:** Removed `.shift(1)` from the `cnt_roll_7` calculation.

**Why this is a problem:** The rolling average now includes the number of bike rentals from the day we are trying to predict. This gives the model information it would not have when making a prediction.

**What happened:** The PostToolUse hook ran after Claude Code made the change. The test `test_rolling_mean_excludes_today` failed because the calculated rolling average was 4389.0 instead of the expected 4414.8571. The test results showed 1 failed and 9 passed.

**Error output:**

```text
FAILED tests/test_leakage.py::test_rolling_mean_excludes_today - assert np.float64(4389.0) == 4414.857142857143 ± 0.00441486
1 failed, 9 passed in 0.53s
```

**How it was fixed:** We added `.shift(1)` back so the rolling average only uses the previous seven days.

**Final result:** After restoring the code, all 10 leakage tests passed. Running `git diff -- src/features.py` showed no remaining changes.

## Fault 2: One-Day Lag Includes Today's Target

**File:** `src/features.py`

**Change made:** Removed `.shift(1)` from the `cnt_lag_1` calculation.

**Why this is a problem:** The feature is supposed to contain yesterday's bike rental count. Without `.shift(1)`, it contains today's count instead. This gives the model the answer it is trying to predict.

**What happened:** The PostToolUse hook ran after Claude Code made the change. Two tests failed: `test_lags_are_true_calendar_lags` and `test_lag_is_not_the_target`. The results showed 2 failed and 8 passed.

**Error output:**

```text
FAILED tests/test_leakage.py::test_lags_are_true_calendar_lags - AssertionError: bad lag at 2011-01-11
FAILED tests/test_leakage.py::test_lag_is_not_the_target - assert not np.True_
2 failed, 8 passed in 0.53s
```

**How it was fixed:** We added `.shift(1)` back so `cnt_lag_1` uses the previous day's count.

**Final result:** After restoring the code, all 10 leakage tests passed.

## Fault 3: Incorrect Seven-Day Lag

**File:** `src/features.py`

**Change made:** Replaced `.shift(LAG_WARMUP)` with `.shift(6)` in the `cnt_lag_7` calculation.

**Why this is a problem:** The feature is supposed to use the bike rental count from seven days earlier. Changing the lag to six days gives the model the wrong historical value.

**What happened:** The existing tests did not detect the mistake. All 10 leakage tests passed, even though the feature was calculated incorrectly.

**How it was fixed:** We restored `.shift(LAG_WARMUP)` to use the correct seven-day lag.

**Final result:** All 10 leakage tests passed after restoring the code.

**What we learned:** Passing all the tests does not always mean the code is correct. A separate test that checks the seven-day lag could help catch this mistake in the future.

## Fault 4: Forbidden Column in Feature List

**File:** `src/features.py`

**Change made:** Added `+ [TARGET]` to the return statement in `feature_columns()`.

**Why this is a problem:** The target column `cnt` is included in the model's input features. This gives the model direct access to the value it is trying to predict.

**What happened:** The PostToolUse hook ran after Claude Code made the change. The test `test_no_forbidden_features` detected that the target column was included in the feature list.

**Error output:**

```text
FAILED tests/test_leakage.py::test_no_forbidden_features
AssertionError: forbidden columns in feature set: {'cnt'}
1 failed, 9 passed in 0.52s
```

**How it was fixed:** We removed `+ [TARGET]` from the return statement so forbidden columns are excluded again.

**Final result:** After restoring the code, all 10 leakage tests passed.

## Fault 5: Removing Weather Encoder Validation

**File:** `src/features.py`

**Change made:** Commented out the `if unknown:` condition and its `raise ValueError` statement in `encode_weather()`.

**Why this is a problem:** Unknown weather labels become NaN instead of raising an error. This can cause rows to be silently removed during feature processing.

**What happened:** The PostToolUse hook did not detect the mistake. The leakage tests do not check how the weather encoder handles unknown labels.

**Test output:**

```text
10 passed in 0.55s
```

**How it was fixed:** We restored the condition and the original `raise ValueError` statement.

**Final result:** After restoring the code, we ran the full test suite.

```text
17 passed in 2.15s
```

**What we learned:** The hook only runs the leakage tests, so it can miss other problems in the code. We did not verify whether the full test suite would have caught this fault before restoring it.

## Fault 6: Writing to the Raw Data Directory

**Directory:** `data/raw/`

**Attempted command:**

```bash
cp README.md data/raw/seeded_fault_test.txt
```

**Why this is a problem:** The raw data directory should remain unchanged throughout the project. Allowing commands to write into this directory could affect the original data and make the results harder to reproduce.

**What happened:** Claude Code denied permission to run the Bash command.

**Error output:**

```text
Not run · permission denied
Reason: Permission to use Bash with command cp README.md data/raw/seeded_fault_test.txt has been denied.
```

**How it was fixed:** No fix was needed because the command was not executed.

**Final result:** The attempted copy was denied.

**What we learned:** The existing permissions prevented this specific Bash command from writing into the raw data directory. This does not guarantee that every other command would also be blocked.
