"""Paired synthetic experiments for preprocessing and validation leakage."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


SEEDS = tuple(range(5))
SCALING_TRAIN = 480
SCALING_END = 640
SERIES_TRAIN = 360
SERIES_DEV_END = 480
SERIES_END = 600
GAP = 8


@dataclass(frozen=True)
class ScalingScore:
    auc: float
    loss: float
    future_fit_rows: int


@dataclass(frozen=True)
class ScalingTrial:
    seed: int
    full_fit: ScalingScore
    train_fit: ScalingScore


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class SeriesTrial:
    seed: int
    random_auc: float
    forward_auc: float
    future_auc: float
    random_exposed: int
    forward_exposed: int
    witness: tuple[int, int] | None


def scaling_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(SCALING_END, 2))
    x[SCALING_TRAIN:, 0] *= 3  # Later periods have more variable spending.
    log_odds = x @ np.array([0.9, -0.6])
    y = rng.random(SCALING_END) < 1 / (1 + np.exp(-log_odds))
    return x, y


def scaling_score(x: np.ndarray, y: np.ndarray, fit_rows: np.ndarray) -> ScalingScore:
    scaler = StandardScaler().fit(x[fit_rows])
    model = LogisticRegression().fit(scaler.transform(x[:SCALING_TRAIN]), y[:SCALING_TRAIN])
    predicted = model.predict_proba(scaler.transform(x[SCALING_TRAIN:]))[:, 1]
    holdout = y[SCALING_TRAIN:]
    return ScalingScore(float(roc_auc_score(holdout, predicted)), float(log_loss(holdout, predicted)),
                        int(np.count_nonzero(fit_rows >= SCALING_TRAIN)))


def scaling_trial(seed: int) -> ScalingTrial:
    x, y = scaling_data(seed)
    return ScalingTrial(seed, scaling_score(x, y, np.arange(SCALING_END)),
                        scaling_score(x, y, np.arange(SCALING_TRAIN)))


def series_data(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    states = np.zeros(SERIES_END)
    for t, shock in enumerate(rng.normal(0, 0.35, size=SERIES_END - 1), start=1):
        states[t] = 0.94 * states[t - 1] + shock
    measured = states + rng.normal(0, 0.85, size=SERIES_END)
    return np.column_stack((measured, np.arange(SERIES_END) / 6)), states > 0


def series_splits(seed: int) -> tuple[Split, Split, Split]:
    order = np.random.default_rng(seed + 10_000).permutation(SERIES_DEV_END)
    random = Split(np.sort(order[:SERIES_TRAIN]), np.sort(order[SERIES_TRAIN + GAP:]))
    forward = Split(np.arange(SERIES_TRAIN), np.arange(SERIES_TRAIN + GAP, SERIES_DEV_END))
    future = Split(np.arange(SERIES_DEV_END), np.arange(SERIES_DEV_END + GAP, SERIES_END))
    return random, forward, future


def near_future_pairs(split: Split) -> tuple[tuple[int, int], ...]:
    """Return one nearby future training row per exposed validation row."""
    positions = np.searchsorted(split.train, split.test, side="right")
    pairs = ((int(t), int(split.train[i])) for t, i in zip(split.test, positions) if i < len(split.train))
    return tuple((test, train) for test, train in pairs if train - test <= GAP)


def series_auc(x: np.ndarray, y: np.ndarray, split: Split) -> float:
    model = KNeighborsClassifier(n_neighbors=5).fit(x[split.train], y[split.train])
    predicted = model.predict_proba(x[split.test])[:, 1]
    return float(roc_auc_score(y[split.test], predicted))


def series_trial(seed: int) -> SeriesTrial:
    x, y = series_data(seed)
    random, forward, future = series_splits(seed)
    random_pairs = near_future_pairs(random)
    forward_pairs = near_future_pairs(forward)
    return SeriesTrial(seed, series_auc(x, y, random), series_auc(x, y, forward), series_auc(x, y, future),
                       len(random_pairs), len(forward_pairs), next(iter(random_pairs), None))
