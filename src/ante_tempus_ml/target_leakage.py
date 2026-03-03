"""Controlled comparison of known past outcomes and the outcome being predicted."""

from dataclasses import dataclass
from math import exp

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


TRAIN_END = 480
BUILD_COUNT = 640


@dataclass(frozen=True)
class OutcomeInput:
    feature: str
    source_build: int
    recorded_at: int
    completed_at: int


@dataclass(frozen=True)
class OutcomeFinding:
    late_by_timestamp: bool
    uses_own_target: bool


@dataclass(frozen=True)
class TargetTrial:
    seed: int
    retrospective_auc: float
    replay_auc: float
    clean_auc: float


def audit_outcome(build: int, prediction_at: int, source: OutcomeInput) -> OutcomeFinding:
    """Check recorded availability and declared target identity separately."""
    return OutcomeFinding(source.recorded_at > prediction_at, source.source_build == build)


def example_sources(build: int) -> tuple[OutcomeInput, ...]:
    return (
        OutcomeInput("Previous build result", build - 1, build, build),
        OutcomeInput("Retry badge", build, build, build + 1),
    )


def build_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=BUILD_COUNT)
    labels = np.zeros(BUILD_COUNT, dtype=bool)
    prior = np.zeros(BUILD_COUNT)
    for build, chance in enumerate(rng.random(BUILD_COUNT)):
        if build:
            prior[build] = 1 if labels[build - 1] else -1
        log_odds = 0.85 * signal[build] + 0.65 * prior[build]
        labels[build] = chance < 1 / (1 + exp(-log_odds))
    # The badge is a noisy encoding generated in hindsight, never a live input.
    badge = np.logical_xor(labels, rng.random(BUILD_COUNT) < 0.15)
    features = np.column_stack((signal, prior, np.zeros(BUILD_COUNT)))
    retrospective = features.copy()
    retrospective[:, 2] = np.where(badge, 1, -1)
    return features, retrospective, labels


def target_trial(seed: int) -> TargetTrial:
    live, retrospective, labels = build_data(seed)
    train = slice(None, TRAIN_END)
    holdout = slice(TRAIN_END, None)
    leaked_model = LogisticRegression().fit(retrospective[train], labels[train])
    clean_model = LogisticRegression().fit(live[train], labels[train])

    def score(model, inputs):
        predicted = model.predict_proba(inputs[holdout])[:, 1]
        return float(roc_auc_score(labels[holdout], predicted))

    return TargetTrial(seed, score(leaked_model, retrospective), score(leaked_model, live), score(clean_model, live))
