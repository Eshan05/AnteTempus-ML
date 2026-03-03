"""One chronological holdout for a timestamp join and source-record replay."""

import argparse
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .join import asof_index
from .provenance import Scenario, Transaction, audit

USERS = 8_000
SEEDS = (11, 29, 47, 71, 103)
LABEL_HORIZON = 30


@dataclass(frozen=True)
class Customer:
    scenario: Scenario
    churned: bool
    activity: float
    injected: str | None


@dataclass(frozen=True)
class Trial:
    seed: int
    churn_rate: float
    test_rows: int
    late_arrivals: int
    late_revisions: int
    timestamp_hits: int
    timestamp_false_flags: int
    honest_timestamp_hits: int
    honest_timestamp_false_flags: int
    source_hits: int
    source_false_flags: int
    retrospective_auc: float
    replay_auc: float
    retrained_auc: float


def population(seed: int, users: int = USERS) -> tuple[Customer, ...]:
    rng = np.random.default_rng(seed)
    start = date(2024, 1, 1)
    train_size = users * 3 // 4
    customers = []
    for i in range(users):
        cutoff = start + timedelta(days=i // 100 + (LABEL_HORIZON + 1 if i >= train_size else 0))
        spend_risk, activity_risk = rng.integers(0, 2, size=2)
        # The two independent risk factors give a marginal churn probability of 30%.
        churned = bool(rng.random() < 0.10 + 0.16 * spend_risk + 0.24 * activity_risk)
        center = round(float(rng.normal(175 - 25 * spend_risk, 16)))
        activity = float(rng.normal(12 - 5 * activity_risk, 2))
        first, second = cutoff - timedelta(days=9), cutoff - timedelta(days=4)
        rows = [Transaction("A-v1", "A", first, first, center - 12),
                Transaction("B-v1", "B", second, second, center + 12)]
        injected = None
        if rng.random() < (0.60 if churned else 0.12):
            arrival = cutoff + timedelta(days=int(rng.integers(1, 8)))
            if rng.random() < 0.5:
                injected = "arrival"
                rows.append(Transaction("C-v1", "C", cutoff - timedelta(days=2), arrival, center - 70))
            else:
                injected = "revision"
                rows.append(Transaction("B-v2", "B", second, arrival, center - 50))
        built_on = cutoff + timedelta(days=10) if injected is not None else cutoff
        case = Scenario(str(i), "Historical spending", tuple(rows), cutoff, cutoff, built_on, bound_window=True)
        customers.append(Customer(case, churned, activity, injected))
    return tuple(customers)


def trial(seed: int, users: int = USERS) -> Trial:
    customers = population(seed, users)
    split = users * 3 // 4
    train, test = slice(0, split), slice(split, None)
    findings = tuple(audit(customer.scenario) for customer in customers)
    batch = np.array([(finding.joined.value, c.activity) for c, finding in zip(customers, findings)])
    live = np.array([(finding.replay_value, c.activity) for c, finding in zip(customers, findings)])
    labels = np.array([c.churned for c in customers])

    def fit(inputs):
        return make_pipeline(StandardScaler(), LogisticRegression()).fit(inputs[train], labels[train])

    def auc(model, inputs):
        return float(roc_auc_score(labels[test], model.predict_proba(inputs[test])[:, 1]))

    retrospective, retrained = fit(batch), fit(live)

    held_out = customers[test]
    truth = np.array([c.injected is not None for c in held_out])
    declared_flags = np.array([f.timestamp_flag for f in findings[test]])
    source_flags = np.array([bool(f.joined.violations) for f in findings[test]])
    # With honest metadata the rebuilt snapshot carries its actual publication day.
    # The previous-day snapshot is still eligible when the candidate is rejected.
    entities = np.arange(len(held_out))
    days = np.array([c.scenario.cutoff.toordinal() for c in held_out])
    built = np.array([c.scenario.built_on.toordinal() for c in held_out])
    honest_match = asof_index(entities, days, np.tile(entities, 2), np.r_[days - 1, days],
                             available_at=np.r_[days - 1, built])
    honest_flags = honest_match < len(held_out)

    def counts(flags):
        return int((flags & truth).sum()), int((flags & ~truth).sum())

    return Trial(seed, float(labels.mean()), len(held_out),
                 sum(c.injected == "arrival" for c in held_out), sum(c.injected == "revision" for c in held_out),
                 *counts(declared_flags), *counts(honest_flags), *counts(source_flags),
                 auc(retrospective, batch), auc(retrospective, live), auc(retrained, live))


def report(trials: tuple[Trial, ...]) -> str:
    def detection(hits, false_flags):
        return sum(getattr(t, hits) for t in trials), sum(getattr(t, false_flags) for t in trials)

    late = sum(t.late_arrivals + t.late_revisions for t in trials)
    held_out = sum(t.test_rows for t in trials)
    rows = ["# Backdated spending features", "",
            "Every transaction occurred before prediction. Some transaction records arrived or were revised later. "
            "A retrospective feature build used them anyway and kept the cutoff as its recorded availability.", "",
            f"Each seed has {USERS:,} users, a 30% marginal churn probability (actual mean: "
            f"{mean(t.churn_rate for t in trials):.2%}), and a chronological 75/25 split. "
            "A 31-day gap separates the training and holdout cutoffs so training labels have time to settle. "
            "The same holdout is scored in every arm.", "", "## Detection", "",
            f"There were {late:,} intentionally contaminated features across {held_out:,} holdout predictions.", "",
            "| Check | Contaminated rows caught | Clean rows flagged |", "| --- | ---: | ---: |"]
    for label, prefix in (("Point-in-time join, backdated metadata", "timestamp"),
                          ("Point-in-time join, honest metadata control", "honest_timestamp"),
                          ("Source audit, backdated metadata", "source")):
        hits, false_flags = detection(f"{prefix}_hits", f"{prefix}_false_flags")
        rows.append(f"| {label} | {hits:,} / {late:,} | {false_flags:,} / {held_out - late:,} |")
    rows.extend(["", "## Churn prediction", "",
                 "The model uses spending and an activity measure already available at prediction. Each model's "
                 "scaler is fit only on its training rows. Replay feeds the retrospectively trained model the "
                 "historically correct features; retraining uses those correct features in both training and scoring.", "",
                 "| Seed | Late arrivals | Late revisions | Backdated offline AUC | Replay AUC | Retrained AUC |",
                 "| ---: | ---: | ---: | ---: | ---: | ---: |"])
    rows.extend(f"| {t.seed} | {t.late_arrivals} | {t.late_revisions} | {t.retrospective_auc:.3f} | "
                f"{t.replay_auc:.3f} | {t.retrained_auc:.3f} |" for t in trials)
    offline, replay, clean = (mean(getattr(t, key) for t in trials) for key in
                              ("retrospective_auc", "replay_auc", "retrained_auc"))
    rows.extend([f"| Mean | | | {offline:.3f} | {replay:.3f} | {clean:.3f} |", "",
                 "```mermaid", "xychart-beta horizontal", '    title "Backdated spending and live replay"',
                 '    x-axis ["Backdated offline", "Historical replay", "Retrained"]',
                 '    y-axis "Mean ROC-AUC (higher is better)" 0 --> 1',
                 f"    bar [{offline:.3f}, {replay:.3f}, {clean:.3f}]", "```", "",
                 "## Scope", "",
                 "This is a synthetic mechanism check. Late reconciliation is deliberately more likely for "
                 "churners (60%) than non-churners (12%). The audit has complete input-version history. "
                 "Neither the contamination rate nor the AUC gap estimates what happens in a real warehouse.", ""])
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/raw/churn-backfill.md"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report(tuple(trial(seed) for seed in SEEDS)), encoding="utf-8", newline="\n")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
