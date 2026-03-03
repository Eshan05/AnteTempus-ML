from datetime import timedelta

from ante_tempus_ml.churn_backfill import LABEL_HORIZON, population, trial
from ante_tempus_ml.provenance import audit


def test_replay_restores_the_original_inputs_for_both_injection_types():
    customers = population(11, 240)
    assert customers == population(11, 240)
    assert customers != population(29, 240)
    assert {c.injected for c in customers} == {None, "arrival", "revision"}
    assert customers[179].scenario.cutoff + timedelta(days=LABEL_HORIZON) < customers[180].scenario.cutoff
    for customer in customers:
        case = customer.scenario
        found = audit(case)
        assert all(row.event_day < case.cutoff for row in case.records)
        assert not found.timestamp_flag
        assert {row.row_id for row in found.replay_inputs} == {"A-v1", "B-v1"}
        assert bool(found.joined.violations) == (customer.injected is not None)
        assert (found.joined.value != found.replay_value) == (customer.injected is not None)


def test_paired_churn_study_distinguishes_hidden_and_disclosed_lateness():
    result = trial(29, 800)
    late = result.late_arrivals + result.late_revisions
    assert 0 < late < result.test_rows == 200
    assert result.timestamp_hits == 0
    assert result.honest_timestamp_hits == result.source_hits == late
    assert result.timestamp_false_flags == result.honest_timestamp_false_flags == result.source_false_flags == 0
    assert all(0 <= score <= 1 for score in (result.retrospective_auc, result.replay_auc, result.retrained_auc))
