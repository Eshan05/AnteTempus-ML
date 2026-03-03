import numpy as np
import pytest

from ante_tempus_ml.join import asof_index, brute_force_index, day_granularity_index, latest_index, leakage_audit, take

FE = np.array([1, 1, 1, 2]); FT = np.array([10, 20, 30, 15])


def test_hand_cases_boundaries_and_unknown_entities():
    ev_e = np.array([1, 1, 1, 1, 2, 3]); ev_t = np.array([9, 10, 25, 99, 14, 50])
    assert list(asof_index(ev_e, ev_t, FE, FT)) == [-1, 0, 1, 2, -1, -1]
    assert list(asof_index(ev_e, ev_t, FE, FT, allow_equal=False)) == [-1, -1, 1, 2, -1, -1]


def test_empty_features_return_missing_rows():
    matches = asof_index([1, 2], [25, 30], [], [])
    assert matches.tolist() == [-1, -1]
    assert take([], matches, fill=-9).tolist() == [-9, -9]
    assert leakage_audit(matches, [], [25, 30])["rows_joined"] == 0


def test_empty_events_return_no_matches():
    assert asof_index([], [], FE, FT).tolist() == []


def test_year_of_nanosecond_timestamps_across_entities():
    year = 365 * 24 * 60 * 60 * 10**9
    entities = np.arange(320)
    got = asof_index(entities, np.full(len(entities), year // 2),
                     np.repeat(entities, 2), np.tile([0, year], len(entities)))
    np.testing.assert_array_equal(got, 2 * entities)


def test_wide_timestamps_preserve_ties_and_ttl():
    lo, hi = -(1 << 63), (1 << 63) - 1
    features = [lo, lo + 1, lo + 1]
    assert asof_index([1, 1], [lo + 1, hi], [1, 1, 1], features, allow_equal=False).tolist() == [0, 2]
    assert asof_index([1], [hi], [1, 1, 1], features, ttl=hi - lo - 1).tolist() == [2]
    assert asof_index([1], [hi], [1, 1, 1], features, ttl=hi - lo - 2).tolist() == [-1]


@pytest.mark.parametrize("timestamp,delay", [((1 << 63) - 1, 1), (-(1 << 63), -1)])
def test_rejects_availability_overflow(timestamp, delay):
    with pytest.raises(ValueError, match="int64"):
        asof_index([1], [0], [1], [timestamp], delay=delay)


def test_delay_and_ttl():
    ev_e = np.array([1, 1]); ev_t = np.array([25, 25])
    assert list(asof_index(ev_e, ev_t, FE, FT, delay=0)) == [1, 1]
    assert list(asof_index(ev_e, ev_t, FE, FT, delay=10)) == [0, 0]          # value at 20 only available at 30
    assert list(asof_index([1], [25], FE, FT, ttl=4)) == [-1]                # 20 is 5 old
    assert list(asof_index([1], [25], FE, FT, ttl=5)) == [1]


def test_per_row_availability_controls_selection_and_ttl():
    availability = [10, 40, 33, 16]
    assert asof_index([1, 1, 1], [25, 33, 40], FE, FT, available_at=availability).tolist() == [0, 2, 1]
    assert asof_index([1], [33], FE, FT, available_at=availability, allow_equal=False).tolist() == [0]
    assert asof_index([1, 1], [36, 37], FE, FT, available_at=availability, ttl=3).tolist() == [2, -1]
    ignored = asof_index([1], [25], FE, FT)
    assert leakage_audit(ignored, FT, [25], available_at=availability)["not_yet_available"] == 1.0


@pytest.mark.parametrize("kwargs", [
    {"available_at": [10]},
    {"available_at": [9, 20, 30, 15]},
    {"available_at": [10, 20, 30, 15], "delay": 1},
])
def test_rejects_invalid_explicit_availability(kwargs):
    with pytest.raises(ValueError, match="available_at"):
        asof_index([1], [25], FE, FT, **kwargs)


@pytest.mark.parametrize("per_row", [False, True])
def test_matches_brute_force_on_random_data(per_row):
    rng = np.random.default_rng(0)
    for _ in range(30):
        fe = rng.integers(0, 6, 40); ft = rng.integers(0, 100, 40)
        ee = rng.integers(0, 8, 30); et = rng.integers(0, 120, 30)
        kw = dict(delay=int(rng.integers(0, 10)), ttl=[None, 15][int(rng.integers(0, 2))], allow_equal=bool(rng.integers(0, 2)))
        if per_row:
            kw.update(delay=0, available_at=ft + rng.integers(0, 10, len(ft)))
        got = asof_index(ee, et, fe, ft, **kw); ref = brute_force_index(ee, et, fe, ft, **kw)
        np.testing.assert_array_equal(got, ref)


def test_matches_pandas_merge_asof():
    import pandas as pd
    rng = np.random.default_rng(1)
    fe = rng.integers(0, 50, 2000); ft = rng.integers(0, 10**5, 2000)
    ee = rng.integers(0, 60, 500); et = rng.integers(0, 10**5, 500)
    idx = asof_index(ee, et, fe, ft)
    left = pd.DataFrame({"e": ee, "t": et, "k": np.arange(500)}).sort_values("t")
    right = pd.DataFrame({"e": fe, "t": ft}).sort_values("t").assign(ft=lambda d: d["t"])
    m = pd.merge_asof(left, right, on="t", by="e").sort_values("k")
    mine = np.where(idx >= 0, ft[np.maximum(idx, 0)], -1)
    assert (mine == m["ft"].fillna(-1).astype(int).values).all()


def test_latest_value_bug_uses_the_future_and_the_audit_sees_it():
    ev_e = np.array([1]); ev_t = np.array([12])
    bad = latest_index(ev_e, FE, FT); good = asof_index(ev_e, ev_t, FE, FT)
    assert FT[bad[0]] == 30 and FT[good[0]] == 10
    assert leakage_audit(bad, FT, ev_t)["future_written"] == 1.0
    assert leakage_audit(good, FT, ev_t)["future_written"] == 0.0


def test_day_granularity_join_leaks_same_day_values():
    day = 86400
    idx = day_granularity_index([1], [5 * day + 100], [1, 1], [5 * day + 50000, 4 * day], day=day)
    assert idx[0] == 0                                                         # the value written LATER the same day
    assert leakage_audit(idx, [5 * day + 50000, 4 * day], [5 * day + 100])["future_written"] == 1.0


def test_audit_flags_unavailable_values_when_delay_is_ignored():
    idx = asof_index([1], [25], FE, FT, delay=0)                               # uses ts 20
    assert leakage_audit(idx, FT, [25], delay=10)["not_yet_available"] == 1.0
    assert leakage_audit(idx, FT, [25], delay=3)["not_yet_available"] == 0.0


def test_take_fills_missing():
    out = take([5.0, 6.0], np.array([1, -1, 0]), fill=-9)
    assert list(out) == [6.0, -9.0, 5.0]
