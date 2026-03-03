import numpy as np
import pytest

from ante_tempus_ml.join import asof_index, brute_force_index, leakage_audit, take

FE = np.array([1, 1, 1, 2]); FT = np.array([10, 20, 30, 15])


def test_unsorted_entity_timelines_keep_input_indices_and_cutoff_ties():
    owners = ["west", "east", "east", "west", "east"]
    updates = [18, 9, 4, 3, 9]
    customers = ["east", "west", "east", "unknown", "west"]
    cutoffs = [9, 17, 3, 500, 18]
    assert asof_index(customers, cutoffs, owners, updates).tolist() == [4, 3, -1, -1, 0]
    assert asof_index(customers, cutoffs, owners, updates, allow_equal=False).tolist() == [2, 3, -1, -1, 3]


def test_missing_numeric_entity_does_not_match_itself():
    assert asof_index([np.nan, 4.0], [7, 7], [np.nan, 4.0], [6, 6]).tolist() == [-1, 1]


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


@pytest.mark.parametrize(("ttl", "expected"), [(0, [0, -1, 1]), (1, [0, 0, 1]), (-1, [-1, -1, -1])])
def test_delay_and_ttl_use_the_same_availability_clock(ttl, expected):
    assert asof_index([8] * 3, [13, 14, 25], [8, 8], [10, 22], delay=3, ttl=ttl).tolist() == expected


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
    log = pd.DataFrame({"account": rng.integers(0, 12, 400), "written": rng.integers(0, 80, 400)})
    log["ready"] = log.written + rng.integers(0, 9, len(log))
    log["row_id"] = log.index
    requests = pd.DataFrame({"account": rng.integers(0, 15, 200), "ready": rng.integers(0, 100, 200)})
    requests["request_id"] = requests.index
    for include_cutoff in (False, True):
        expected = pd.merge_asof(requests.sort_values("ready", kind="stable"),
                                 log.sort_values("ready", kind="stable"), on="ready", by="account", tolerance=7,
                                 allow_exact_matches=include_cutoff).sort_values("request_id")
        selected = asof_index(requests.account, requests.ready, log.account, log.written,
                              available_at=log.ready, ttl=7, allow_equal=include_cutoff)
        np.testing.assert_array_equal(selected, expected.row_id.fillna(-1).astype(int))


def test_audit_distinguishes_future_writes_from_late_delivery():
    assert leakage_audit([0, 1, 2, -1], [12, 28, 18], [25, 25, 18, 100], available_at=[26, 31, 18]) == {
        "rows_joined": 3, "future_written": 1 / 3, "not_yet_available": 2 / 3,
    }


def test_take_keeps_the_order_and_fills_each_unmatched_row():
    np.testing.assert_array_equal(take([4, 9, 2], [2, -1, 2, 0, -1], fill=-7), [2, -7, 2, 4, -7])
