import numpy as np

from ante_tempus_ml.validation import (
    GAP,
    SCALING_END,
    SCALING_TRAIN,
    SEEDS,
    near_future_pairs,
    scaling_trial,
    series_splits,
    series_trial,
)


def test_scaler_protocols_differ_in_future_fit_exposure_not_holdout_size():
    trial = scaling_trial(0)
    assert trial.full_fit.future_fit_rows == SCALING_END - SCALING_TRAIN
    assert trial.train_fit.future_fit_rows == 0
    assert all(0 <= score.auc <= 1 and np.isfinite(score.loss) for score in (trial.full_fit, trial.train_fit))


def test_development_folds_are_matched_and_cannot_use_the_future_holdout():
    for seed in SEEDS:
        random, forward, future = series_splits(seed)
        assert len(random.train) == len(forward.train)
        assert len(random.test) == len(forward.test) == len(future.test)
        for split in (random, forward, future):
            assert not np.intersect1d(split.train, split.test).size
        for split in (random, forward):
            assert set(split.train) | set(split.test) <= set(future.train)
        for split in (forward, future):
            assert split.test.min() - split.train.max() == GAP + 1


def test_random_validation_has_future_neighbors_and_forward_does_not():
    random, forward, future = series_splits(0)
    witnesses = near_future_pairs(random)
    assert witnesses
    assert len({test for test, _ in witnesses}) == len(witnesses)
    assert all(test in random.test and train in random.train and 0 < train - test <= GAP for test, train in witnesses)
    assert near_future_pairs(forward) == near_future_pairs(future) == ()
    trial = series_trial(0)
    assert trial.witness == witnesses[0]
    assert trial.random_exposed == len(witnesses)
    assert trial.forward_exposed == 0
    assert all(0 <= score <= 1 for score in (trial.random_auc, trial.forward_auc, trial.future_auc))
