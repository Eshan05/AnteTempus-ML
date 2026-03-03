"""Controlled audit of spending features and fitted statistics with known inputs."""

from dataclasses import dataclass, replace
from datetime import date, timedelta
from math import sqrt
from statistics import mean

from sklearn.preprocessing import StandardScaler

from .join import asof_index


@dataclass(frozen=True)
class Transaction:
    row_id: str
    transaction_id: str
    event_day: date
    available_day: date
    dollars: int


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    records: tuple[Transaction, ...]
    cutoff: date
    declared_available: date
    built_on: date
    bound_window: bool
    prediction_dollars: int | None = None


@dataclass(frozen=True)
class Fit:
    mean: float
    scale: float
    z_score: float


@dataclass(frozen=True)
class JoinedFeature:
    is_candidate: bool
    available_day: date
    value: float | None
    violations: tuple[Transaction, ...]


@dataclass(frozen=True)
class Audit:
    timestamp_flag: bool
    batch_inputs: tuple[Transaction, ...]
    replay_inputs: tuple[Transaction, ...]
    violations: tuple[Transaction, ...]
    batch_value: float | None
    replay_value: float | None
    joined: JoinedFeature
    batch_fit: Fit | None = None
    replay_fit: Fit | None = None


def materialize(case: Scenario) -> tuple[Transaction, ...]:
    """Rebuild the month from the warehouse as seen by the feature job."""
    latest = {}
    for row in sorted(case.records, key=lambda r: (r.available_day, r.row_id), reverse=True):
        if row.available_day > case.built_on:
            continue
        latest.setdefault(row.transaction_id, row)
    return tuple(row for row in latest.values() if not case.bound_window or row.event_day <= case.cutoff)


def replay(records: tuple[Transaction, ...], cutoff: date) -> tuple[Transaction, ...]:
    """Independently replay source arrivals, then run the historical window."""
    online = {}
    for row in sorted(records, key=lambda r: (r.available_day, r.row_id)):
        if row.available_day > cutoff:
            break
        online[row.transaction_id] = row
    return tuple(row for row in online.values() if row.event_day <= cutoff)


def batch_scale(rows: tuple[Transaction, ...], amount: int) -> Fit | None:
    if not rows:
        return None
    scaler = StandardScaler().fit([[row.dollars] for row in rows])
    z_score = scaler.transform([[amount]])[0, 0]
    return Fit(float(scaler.mean_[0]), float(scaler.scale_[0]), float(z_score))


def replay_scale(rows: tuple[Transaction, ...], amount: int) -> Fit | None:
    if not rows:
        return None
    center = sum(row.dollars for row in rows) / len(rows)
    scale = sqrt(sum((row.dollars - center) ** 2 for row in rows) / len(rows)) or 1.0
    return Fit(center, scale, (amount - center) / scale)


def audit(case: Scenario) -> Audit:
    # The demo offers a previous-day snapshot and a candidate dated at the cutoff.
    previous_day = case.cutoff - timedelta(days=1)
    day = case.cutoff.toordinal()
    previous = previous_day.toordinal()
    selected, = asof_index([0], [day], [0, 0], [previous, day],
                           available_at=[previous, case.declared_available.toordinal()])
    batch = materialize(case)
    historical = replay(case.records, case.cutoff)
    late = tuple(row for row in batch if row.available_day > case.cutoff or row.event_day > case.cutoff)
    fitted = reference = None
    if case.prediction_dollars is not None:
        fitted = batch_scale(batch, case.prediction_dollars)
        reference = replay_scale(historical, case.prediction_dollars)
        batch_value = fitted.z_score if fitted else None
        replay_value = reference.z_score if reference else None
    else:
        batch_value = mean(row.dollars for row in batch) if batch else None
        replay_value = mean(row.dollars for row in historical) if historical else None

    if selected == 1:
        joined = JoinedFeature(True, case.declared_available, batch_value, late)
    else:
        prior = materialize(replace(case, cutoff=previous_day, built_on=previous_day, bound_window=True))
        if case.prediction_dollars is not None:
            prior_fit = batch_scale(prior, case.prediction_dollars)
            prior_value = prior_fit.z_score if prior_fit else None
        else:
            prior_value = mean(row.dollars for row in prior) if prior else None
        joined = JoinedFeature(False, previous_day, prior_value, ())
    return Audit(not joined.is_candidate, batch, historical, late, batch_value, replay_value, joined, fitted, reference)


def scenarios() -> tuple[Scenario, ...]:
    def day(n: int) -> date:
        return date(2026, 11, n)

    a = Transaction("T1-v1", "T1", day(4), day(4), 20)
    b = Transaction("T2-v1", "T2", day(8), day(8), 40)
    future = Transaction("T3-v1", "T3", day(30), day(30), 90)
    revision = Transaction("T2-v2", "T2", day(8), day(12), 70)
    cutoff = day(10)
    return (
        Scenario("hidden", "Future transaction in a backdated average", (a, b, future), cutoff, cutoff, day(30), False),
        Scenario("declared", "Same average with an honest availability date", (a, b, future), cutoff, day(30), day(30), False),
        Scenario("revision", "Late correction to an earlier transaction", (a, b, revision), cutoff, cutoff, day(12), True),
        Scenario("clean", "Clean historical average", (a, b, future, revision), cutoff, cutoff, cutoff, True),
        Scenario("scaler-leak", "Scaler fit on the full dataset", (a, b, future), cutoff, cutoff, day(30), False,
                 prediction_dollars=50),
        Scenario("scaler-clean", "Scaler fit before prediction", (a, b, future), cutoff, cutoff, cutoff, True,
                 prediction_dollars=50),
    )
