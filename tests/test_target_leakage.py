from dataclasses import replace

import numpy as np

from ante_tempus_ml.target_leakage import TRAIN_END, audit_outcome, build_data, example_sources, target_trial


def test_matching_target_fails_even_with_a_backdated_source():
    previous, own = example_sources(TRAIN_END)
    safe = audit_outcome(TRAIN_END, TRAIN_END, previous)
    leaky = audit_outcome(TRAIN_END, TRAIN_END, own)
    assert not safe.late_by_timestamp and not safe.uses_own_target
    assert not leaky.late_by_timestamp and leaky.uses_own_target

    honestly_recorded = audit_outcome(TRAIN_END, TRAIN_END, replace(own, recorded_at=own.completed_at))
    assert honestly_recorded.late_by_timestamp and honestly_recorded.uses_own_target
    delayed_prior = audit_outcome(TRAIN_END, TRAIN_END, replace(previous, recorded_at=TRAIN_END + 1))
    assert delayed_prior.late_by_timestamp and not delayed_prior.uses_own_target


def test_retrospective_input_changes_only_the_target_derived_feature():
    live, retrospective, labels = build_data(0)
    np.testing.assert_array_equal(live[:, :2], retrospective[:, :2])
    np.testing.assert_array_equal(live[1:, 1], np.where(labels[:-1], 1, -1))
    assert not live[:, 2].any()
    match = ((retrospective[:, 2] > 0) == labels).mean()
    assert 0.75 < match < 0.95
    trial = target_trial(0)
    assert trial.retrospective_auc > trial.replay_auc
    assert 0 <= trial.clean_auc <= 1
