"""Build 30-day churn labels from a complete slice of cancellation history."""

from bisect import bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from .join import asof_index


HORIZON = timedelta(days=30)


@dataclass(frozen=True)
class Prediction:
    """An active subscription at prediction and the inclusive end of label-data coverage."""

    customer: str
    at: date
    observed_through: date


def churn_labels(
    predictions: Iterable[Prediction], cancellations: Mapping[str, Iterable[date]]
) -> tuple[bool | None, ...]:
    """Look forward for a cancellation in (at, at + 30 days]; None means incomplete coverage."""
    by_customer = {customer: sorted(events) for customer, events in cancellations.items()}
    labels = []
    for prediction in predictions:
        deadline = prediction.at + HORIZON
        if prediction.observed_through < deadline:
            labels.append(None)
            continue
        events = by_customer.get(prediction.customer, ())
        next_event = bisect_right(events, prediction.at)
        labels.append(next_event < len(events) and events[next_event] <= deadline)
    return tuple(labels)


@dataclass(frozen=True)
class ExampleRow:
    prediction: Prediction
    cancellation: date | None
    tickets: int
    label: bool | None


def example() -> tuple[ExampleRow, ...]:
    at = date(2026, 1, 10)
    cohort = (
        ("A", 12, 30),
        ("B", 30, 30),
        ("C", 31, 35),
        ("D", None, 30),
        ("E", None, 18),
    )
    customers = [customer for customer, _, _ in cohort]
    predictions = [Prediction(customer, at, at + timedelta(days=coverage)) for customer, _, coverage in cohort]
    cancellations = {customer: (at + timedelta(days=day),) for customer, day, _ in cohort if day is not None}

    day = at.toordinal()
    n = len(cohort)
    # Yesterday's newer count is not published until tomorrow.
    updates = [2] * n + [7] * n
    indices = asof_index(customers, [day] * n, customers * 2,
                         [day - 2] * n + [day - 1] * n,
                         available_at=[day - 2] * n + [day + 1] * n)
    labels = churn_labels(predictions, cancellations)
    return tuple(ExampleRow(p, cancellations.get(p.customer, (None,))[0], updates[index], label)
                 for p, index, label in zip(predictions, indices, labels, strict=True))
