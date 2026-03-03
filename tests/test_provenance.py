from dataclasses import replace
from datetime import timedelta

import pytest

from ante_tempus_ml.provenance import Transaction, audit, scenarios
from ante_tempus_ml.provenance_report import main


@pytest.mark.parametrize("key,timestamp_flag,witness,batch_value", [
    ("hidden", False, "T3-v1", 50),
    ("declared", True, "T3-v1", 50),
    ("revision", False, "T2-v2", 45),
    ("clean", False, None, 30),
])
def test_audit_agrees_with_historical_replay(key, timestamp_flag, witness, batch_value):
    case = next(case for case in scenarios() if case.key == key)
    result = audit(case)
    assert result.timestamp_flag is timestamp_flag
    assert result.batch_value == batch_value
    assert result.replay_value == 30
    assert [row.row_id for row in result.violations] == ([witness] if witness else [])
    assert {row.row_id for row in result.replay_inputs} == {"T1-v1", "T2-v1"}
    assert result.joined.is_candidate is not timestamp_flag
    assert result.joined.value == (30 if timestamp_flag else batch_value)
    assert result.joined.violations == (() if timestamp_flag else result.violations)


def test_fallback_is_a_previous_snapshot_not_the_cutoff_replay():
    case = next(case for case in scenarios() if case.key == "declared")
    at_cutoff = Transaction("T4-v1", "T4", case.cutoff, case.cutoff, 60)
    result = audit(replace(case, records=case.records + (at_cutoff,)))
    assert result.replay_value == 40
    assert result.joined.available_day == case.cutoff - timedelta(days=1)
    assert result.joined.value == 30
    assert result.joined.violations == ()


def test_revision_is_applied_once_it_is_available():
    case = next(case for case in scenarios() if case.key == "revision")
    later = replace(case, cutoff=case.built_on, declared_available=case.built_on)
    result = audit(later)
    assert result.replay_value == result.batch_value == 45
    assert {row.row_id for row in result.replay_inputs} == {"T1-v1", "T2-v2"}
    assert result.violations == ()


def test_normalization_leaks_through_fitted_statistics_not_the_prediction_input():
    cases = {case.key: case for case in scenarios()}
    leaky = cases["scaler-leak"]
    clean = cases["scaler-clean"]
    assert leaky.prediction_dollars == clean.prediction_dollars == 50

    result = audit(leaky)
    control = audit(clean)
    assert not result.timestamp_flag
    assert [row.row_id for row in result.violations] == ["T3-v1"]
    assert result.batch_fit.mean == 50
    assert result.batch_fit.scale == pytest.approx(29.4392028878)
    assert result.batch_fit.z_score == 0
    assert result.replay_fit.mean == 30
    assert result.replay_fit.scale == 10
    assert result.replay_fit.z_score == 2
    assert control.batch_fit == control.replay_fit == result.replay_fit
    assert control.violations == ()


def test_normalization_replay_matches_scaler_on_constant_fit_inputs():
    case = next(case for case in scenarios() if case.key == "scaler-clean")
    case = replace(case, records=tuple(replace(row, dollars=40) for row in case.records))
    result = audit(case)
    assert result.batch_fit == result.replay_fit
    assert result.replay_fit.scale == 1
    assert result.replay_fit.z_score == 10


def test_report_contains_clickable_witness_and_all_controls(tmp_path):
    path = tmp_path / "report.html"
    main(["--out", str(path)])
    html = path.read_text(encoding="utf-8")
    assert "<title>Ante Tempus ML | Spending feature audit</title>" in html
    assert '<details id="hidden" open>' in html
    assert '<a href="#hidden-T3-v1">T3-v1</a>' in html
    assert 'id="hidden-T3-v1" class="record witness"' in html
    assert '<details id="declared">' in html
    assert 'As-of join: <strong>Previous update</strong> (2026-11-09)' in html
    assert 'As-of join: <strong>Cutoff update</strong> (2026-11-10)' in html
    assert '<details id="revision">' in html
    assert '<details id="clean">' in html
    assert '<a href="#normalization">Normalization</a>' in html
    assert '<a href="#scaler-leak-T3-v1">T3-v1</a>' in html
    assert "Fit mean $50.00 &middot; scale $29.44" in html
    assert "Fit mean $30.00 &middot; scale $10.00" in html
    assert '<details id="scaler-clean">' in html
    assert '<a href="#validation">Validation studies</a>' in html
    assert "Training-only fit" in html and "Forward validation" in html
    assert "ROC-AUC by seed" in html
    assert '<a href="#target-leakage">Target leakage</a>' in html
    assert '<a href="#outcome-480">this same build</a>' in html
    assert 'id="outcome-480" class="record witness"' in html
    assert 'id="outcome-479" class="record"' in html
    assert '<a href="#churn">Churn labels</a>' in html
    assert '<td>Day 30</td><td>30 days</td><td>Churned</td>' in html
    assert '<td>None observed</td><td>18 days</td><td>Excluded</td>' in html


def test_report_refuses_to_overwrite_a_markdown_result(tmp_path):
    path = tmp_path / "report.md"
    path.write_text("Baseline", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["--out", str(path)])
    assert path.read_text(encoding="utf-8") == "Baseline"
