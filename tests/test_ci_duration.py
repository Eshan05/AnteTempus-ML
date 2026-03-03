from datetime import datetime, timedelta, timezone

import pytest

from ante_tempus_ml import ci_source
from ante_tempus_ml.ci_duration import Job, chronological_split, features, score_window, study, with_history
from ante_tempus_ml.provenance_report import ci_html, ci_section


ORIGIN = datetime(2026, 11, 21, tzinfo=timezone.utc)


def job(job_id, run_id, start, end, name="Test", failed=False):
    return Job(job_id, run_id, name, ORIGIN + timedelta(seconds=start), ORIGIN + timedelta(seconds=end), failed)


def test_history_rejects_overlap_equal_timestamp_and_same_run():
    known = job(1, 1, 0, 5)
    tied = job(2, 2, 6, 20)
    sibling = job(3, 3, 10, 12)
    current = job(4, 3, 20, 40)
    different_task = job(5, 2, 11, 13, name="Lint")
    row = next(r for r in with_history([current, known, tied, sibling, different_task]) if r.job == current)
    assert row.previous_completed == known
    assert row.overlapping_predecessor == tied
    assert set(features(row, 0)) == {"job"}
    assert set(features(row, 1)) == {"job", "prior_missing", "prior_duration"}
    assert set(features(row, 2)) == {"job", "prior_missing", "prior_duration", "final_duration"}


def fixture_jobs():
    return [job(run * 2 + i, run, run * 10 + i, run * 10 + (25 if run == 7 and i == 0 else 5),
                name=("Linux", "Windows")[i], failed=(run + i) % 2 == 0)
            for run in range(10) for i in range(2)]


def fixture_dataset(cohort):
    raw = [{"id": j.id, "run_id": j.run_id, "name": j.name, "started_at": j.started.isoformat(),
            "completed_at": j.completed.isoformat(), "conclusion": "failure" if j.failed else "success"}
           for j in fixture_jobs()]
    return {"repo": cohort.repo, "workflow": cohort.workflow, "created_window": cohort.created,
            "attempt": 1, "jobs": raw}


def test_run_level_split_excludes_training_labels_unavailable_at_cutoff():
    rows = with_history(fixture_jobs())
    train, test, cutoff, purged, runs = chronological_split(rows)
    assert {r.job.run_id for r in test} == {8, 9}
    assert all(r.job.run_id < 8 and r.job.completed < cutoff for r in train)
    assert cutoff == ORIGIN + timedelta(seconds=80)
    assert (len(train), len(test), purged, runs) == (15, 4, 1, 2)

    test_ids = []
    for window in range(1, 5):
        earlier, holdout, fence, _, _ = chronological_split(rows, window)
        assert all(row.job.completed < fence for row in earlier)
        test_ids.extend(row.job.id for row in holdout)
    assert len(test_ids) == len(set(test_ids)) == 16


def test_one_class_window_has_no_auc_but_keeps_log_loss():
    rows = with_history(fixture_jobs())
    train, test, _, _, _ = chronological_split(rows)
    scores = score_window(train, [row for row in test if not row.job.failed])
    assert all(s.auc is None and s.ap is None and s.loss > 0 for s in scores)


def test_study_and_report_use_shared_holdout_and_real_witness_links():
    dataset = fixture_dataset(ci_source.PANDAS)
    raw = dataset["jobs"]
    raw.append({**raw[0], "id": 99, "conclusion": "cancelled"})
    result = study(raw)
    assert result.excluded == 1
    assert sum(window.holdout_failures for window in result.windows) == 8
    assert result.own_unavailable == sum(window.holdout for window in result.windows) == 16
    assert len(result.scores) == 3 and all(score.auc is not None and 0 <= score.auc <= 1 for score in result.scores)
    report = ci_html(dataset)
    assert "Mean window ROC-AUC" in report
    assert "Earlier start, not yet complete" in report
    assert "https://github.com/pandas-dev/pandas/actions/runs/8/job/16" in report


def test_report_keeps_cohorts_and_witness_links_separate():
    pandas, scipy = (fixture_dataset(c) for c in (ci_source.PANDAS, ci_source.SCIPY))
    section = ci_section([pandas, scipy])
    assert section.count('id="ci-duration"') == 1
    assert 'id="ci-pandas">pandas Unit Tests' in section
    assert 'id="ci-scipy">SciPy Linux tests' in section
    assert "https://github.com/scipy/scipy/actions/runs/8/job/16" in section
    assert section.count("Mean window ROC-AUC") == 2
    with pytest.raises(ValueError, match="at most one"):
        ci_section([pandas, pandas])


@pytest.mark.parametrize("last_id", [100, 99])
def test_collect_checks_paginated_ids(monkeypatch, last_id):
    def fake(endpoint):
        assert "per_page=100" in endpoint
        ids = [last_id] if "page=2" in endpoint else list(range(100))
        return {"total_count": 101, "workflow_runs": [{"id": i} for i in ids]}

    monkeypatch.setattr(ci_source, "gh", fake)
    if last_id == 99:
        with pytest.raises(ValueError, match="duplicate"):
            ci_source.list_records("runs", "workflow_runs")
    else:
        assert [r["id"] for r in ci_source.list_records("runs", "workflow_runs")] == list(range(101))


def test_collector_pins_first_attempt_and_only_stores_analysis_fields(monkeypatch):
    def fake(endpoint):
        if "/workflows/" in endpoint:
            assert "2026-11-21..2026-11-23" in endpoint and "event=pull_request" in endpoint
            return {"total_count": 1, "workflow_runs": [{"id": 123, "run_attempt": 3}]}
        assert endpoint.endswith("/runs/123/attempts/1/jobs?per_page=100")
        return {"total_count": 1, "jobs": [{"id": 42, "name": "Test", "started_at": "a", "completed_at": "b",
                                            "conclusion": "success", "runner_name": "not collected"}]}

    monkeypatch.setattr(ci_source, "gh", fake)
    assert ci_source.collect()["jobs"] == [{"run_id": 123, "id": 42, "name": "Test", "started_at": "a",
                                            "completed_at": "b", "conclusion": "success"}]
