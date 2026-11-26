"""The tests that make the agent honest.

test_no_forbidden_features is the single most valuable test in this repo:
casual + registered == cnt exactly, so any model handed those columns
scores R^2 = 1.0 and looks brilliant while having learned nothing.
"""
import pandas as pd
import pytest

from src.config import FORBIDDEN, TARGET
from src.features import build_features, feature_columns
from src.ingest import load_raw


@pytest.fixture(scope="module")
def feats():
    return build_features(load_raw())


def test_leakage_identity_holds(feats):
    """Documents *why* casual and registered are forbidden."""
    assert (feats.casual + feats.registered == feats[TARGET]).all()


def test_no_forbidden_features(feats):
    """Guards the gate, not the data: feature_columns() excludes FORBIDDEN by
    construction, so this cannot discover a *new* leaky column. What it
    catches is someone editing feature_columns() or FORBIDDEN -- which is
    exactly the fault the Step 7 hook demonstrates."""
    cols = set(feature_columns(feats))
    assert not (cols & FORBIDDEN), f"forbidden columns in feature set: {cols & FORBIDDEN}"


def test_lags_are_true_calendar_lags(feats):
    """cnt_lag_1 for date d must equal cnt on d-1, in calendar days."""
    by_date = feats.set_index("dteday")[TARGET]
    checked = 0
    for date, lag in feats.set_index("dteday").cnt_lag_1.items():
        prev = date - pd.Timedelta(days=1)
        if prev in by_date.index:
            assert lag == pytest.approx(by_date[prev]), f"bad lag at {date.date()}"
            checked += 1
    assert checked > 600, "too few rows actually verified"


def test_lag_is_not_the_target(feats):
    assert not (feats.cnt_lag_1 == feats[TARGET]).all()


def test_rolling_mean_excludes_today(feats):
    """cnt_roll_7 must be the mean of the 7 calendar days *before* today."""
    by_date = feats.set_index("dteday")[TARGET]
    date = feats.dteday.iloc[200]
    window = [date - pd.Timedelta(days=k) for k in range(1, 8)]
    if all(d in by_date.index for d in window):
        expected = sum(by_date[d] for d in window) / 7
        assert feats.set_index("dteday").cnt_roll_7[date] == pytest.approx(expected)


def test_no_missing_values(feats):
    assert feats[feature_columns(feats)].isna().sum().sum() == 0


def test_dates_are_sorted_and_unique(feats):
    assert feats.dteday.is_monotonic_increasing
    assert feats.dteday.is_unique



def test_weather_severity_roll_3_exists(feats):
    """The new feature must be included in the model's feature list."""
    assert "weather_severity_roll_3" in feature_columns(feats)


def test_weather_severity_roll_3_calculation_is_correct(feats):
    """Use the previous three calendar days, excluding today."""
    by_date = feats.set_index("dteday")
    date = pd.Timestamp("2012-05-15")

    assert date in by_date.index

    previous_days = [
        date - pd.Timedelta(days=k)
        for k in (3, 2, 1)
    ]

    assert all(d in by_date.index for d in previous_days)

    expected = sum(
        by_date.loc[d, "weather_severity"]
        for d in previous_days
    ) / 3

    actual = by_date.loc[date, "weather_severity_roll_3"]

    assert actual == pytest.approx(expected)


def test_weather_severity_roll_3_with_missing_calendar_date():
    """A missing calendar day must not be treated as an available observation."""
    raw = load_raw().copy()
    raw["dteday"] = pd.to_datetime(raw["dteday"])

    missing_date = pd.Timestamp("2012-05-13")
    affected_date = pd.Timestamp("2012-05-15")
    unaffected_date = pd.Timestamp("2012-05-21")

    raw = raw.loc[raw["dteday"] != missing_date].copy()

    feats = build_features(raw)
    by_date = feats.set_index("dteday")

    assert "weather_severity_roll_3" in feats.columns

    assert affected_date not in by_date.index
    assert unaffected_date in by_date.index

    previous_days = [
        unaffected_date - pd.Timedelta(days=k)
        for k in (1, 2, 3)
    ]

    expected = raw.set_index("dteday").loc[
        previous_days, "weather"
    ].map({"GOOD": 0, "MISTY": 1, "BAD": 2}).mean()

    actual = by_date.loc[unaffected_date, "weather_severity_roll_3"]

    assert actual == pytest.approx(expected)



def test_seven_day_calendar_lag(feats):
    raw = load_raw().set_index("dteday")[TARGET]

    for date, lag in feats.set_index("dteday").cnt_lag_7.items():
        previous_date = date - pd.Timedelta(days=7)
        assert previous_date in raw.index
        assert lag == pytest.approx(raw.loc[previous_date])


def test_unknown_weather_label_raises_error():
    from src.features import encode_weather

    weather = pd.Series(["GOOD", "UNKNOWN"])

    with pytest.raises(ValueError, match="unmapped weather labels"):
        encode_weather(weather)
