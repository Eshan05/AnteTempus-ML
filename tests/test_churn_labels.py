from datetime import date, timedelta

import pytest

from ante_tempus_ml.churn_labels import Prediction, churn_labels, example


AT = date(2026, 1, 10)


@pytest.mark.parametrize("events,coverage,expected", [
    ([30], 30, True),
    ([31], 35, False),
    ([-2, 0], 30, False),
    ([], 30, False),
    ([31, 12, 0, -2], 35, True),
    ([], 29, None),
    ([12], 18, None),
])
def test_forward_labels_require_a_complete_horizon(events, coverage, expected):
    predictions = [Prediction("A", AT, AT + timedelta(days=coverage))]
    cancellations = {"A": [AT + timedelta(days=d) for d in events], "B": [AT + timedelta(days=1)]}
    assert churn_labels(predictions, cancellations) == (expected,)


def test_example_joins_observed_features_and_forward_labels_separately():
    rows = example()
    assert [row.tickets for row in rows] == [2, 2, 2, 2, 2]
    assert [row.label for row in rows] == [True, True, False, False, None]
