"""Check when CI job durations become available and compare matched holdout scores."""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from math import log1p
from statistics import mean

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline


@dataclass(frozen=True)
class Job:
    id: int
    run_id: int
    name: str
    started: datetime
    completed: datetime
    failed: bool

    @property
    def seconds(self) -> float:
        return (self.completed - self.started).total_seconds()


@dataclass(frozen=True)
class TimedRow:
    job: Job
    previous_completed: Job | None
    overlapping_predecessor: Job | None


@dataclass(frozen=True)
class Score:
    name: str
    auc: float | None
    ap: float | None
    loss: float


@dataclass(frozen=True)
class Window:
    train: int
    purged: int
    holdout: int
    holdout_runs: int
    holdout_failures: int
    cutoff: datetime
    scores: tuple[Score, ...]


@dataclass(frozen=True)
class Study:
    eligible: int
    excluded: int
    windows: tuple[Window, ...]
    scores: tuple[Score, ...]
    with_history: int
    with_overlap: int
    own_unavailable: int
    witness: TimedRow


def time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("CI times must carry a UTC offset")
    return parsed


def eligible_jobs(raw: list[dict]) -> list[Job]:
    jobs = []
    seen = set()
    for row in raw:
        if row.get("conclusion") not in ("success", "failure") or not row.get("name"):
            continue
        try:
            job = Job(int(row["id"]), int(row["run_id"]), row["name"], time(row["started_at"]),
                      time(row["completed_at"]), row["conclusion"] == "failure")
        except (AttributeError, KeyError, TypeError, ValueError):
            continue
        if job.seconds > 0 and job.id not in seen:
            jobs.append(job)
            seen.add(job.id)
    return jobs


def with_history(jobs: list[Job]) -> list[TimedRow]:
    """Replay earlier workflow runs, using only completions strictly before start."""
    groups = defaultdict(list)
    for job in jobs:
        groups[job.name].append(job)
    rows = []
    for group in groups.values():
        started = sorted(group, key=lambda j: (j.started, j.id))
        completed = sorted(group, key=lambda j: (j.completed, j.id))
        available = []
        cursor = 0
        for i, job in enumerate(started):
            while cursor < len(completed) and completed[cursor].completed < job.started:
                available.append(completed[cursor])
                cursor += 1
            prior = next((j for j in reversed(available) if j.run_id != job.run_id), None)
            predecessor = next((j for j in reversed(started[:i])
                                if j.run_id != job.run_id and j.started < job.started), None)
            overlap = predecessor if predecessor and predecessor.completed >= job.started else None
            rows.append(TimedRow(job, prior, overlap))
    return sorted(rows, key=lambda row: (row.job.started, row.job.id))


def chronological_split(rows: list[TimedRow], window: int = 4) -> tuple[list[TimedRow], list[TimedRow], datetime, int, int]:
    starts = {}
    for row in rows:
        run_id = row.job.run_id
        starts[run_id] = min(starts.get(run_id, row.job.started), row.job.started)
    if len(starts) < 5 or not 1 <= window <= 4:
        raise ValueError("Forward windows 1-4 require at least five eligible workflow runs")
    order = sorted(starts, key=lambda run_id: (starts[run_id], run_id))
    left, right = len(order) * window // 5, len(order) * (window + 1) // 5
    earlier, future = set(order[:left]), set(order[left:right])
    cutoff = min(starts[run_id] for run_id in future)
    holdout = [row for row in rows if row.job.run_id in future]
    candidates = [row for row in rows if row.job.run_id in earlier]
    train = [row for row in candidates if row.job.completed < cutoff]
    return train, holdout, cutoff, len(candidates) - len(train), len(future)


def features(row: TimedRow, arm: int) -> dict:
    result = {"job": row.job.name}
    if arm >= 1:
        result["prior_missing"] = row.previous_completed is None
        if row.previous_completed is not None:
            result["prior_duration"] = log1p(row.previous_completed.seconds)
    if arm == 2:
        result["final_duration"] = log1p(row.job.seconds)
    return result


def score_window(train: list[TimedRow], test: list[TimedRow]) -> tuple[Score, ...]:
    y_train = [row.job.failed for row in train]
    y_test = [row.job.failed for row in test]
    if len(set(y_train)) != 2:
        raise ValueError("Training must contain both successes and failures")
    both = len(set(y_test)) == 2

    scores = []
    for arm, name in enumerate(("No duration", "Prior completed duration", "Current duration (leaky)")):
        model = make_pipeline(DictVectorizer(), LogisticRegression(max_iter=1000))
        model.fit([features(row, arm) for row in train], y_train)
        prediction = model.predict_proba([features(row, arm) for row in test])[:, 1]
        scores.append(Score(name, float(roc_auc_score(y_test, prediction)) if both else None,
                            float(average_precision_score(y_test, prediction)) if both else None,
                            float(log_loss(y_test, prediction, labels=[False, True]))))
    return tuple(scores)


def study(raw: list[dict]) -> Study:
    rows = with_history(eligible_jobs(raw))
    windows = []
    evaluated = []
    for window in range(1, 5):
        train, test, cutoff, purged, test_runs = chronological_split(rows, window)
        evaluated.extend(test)
        windows.append(Window(len(train), purged, len(test), test_runs, sum(r.job.failed for r in test),
                              cutoff, score_window(train, test)))
    scores = []
    for arm in range(3):
        arms = [window.scores[arm] for window in windows]
        aucs = [score.auc for score in arms if score.auc is not None]
        aps = [score.ap for score in arms if score.ap is not None]
        scores.append(Score(arms[0].name, mean(aucs) if aucs else None, mean(aps) if aps else None,
                            sum(score.loss * w.holdout for score, w in zip(arms, windows)) / len(evaluated)))
    witness = next((row for row in evaluated if row.previous_completed and row.overlapping_predecessor), evaluated[0])
    return Study(len(rows), len(raw) - len(rows), tuple(windows), tuple(scores),
                 sum(r.previous_completed is not None for r in evaluated),
                 sum(r.overlapping_predecessor is not None for r in evaluated),
                 sum(r.job.completed > r.job.started for r in evaluated), witness)
