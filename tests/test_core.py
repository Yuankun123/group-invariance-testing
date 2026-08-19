import math

import numpy as np
import pytest

from group_invariance_testing import (
    EuclideanSpace,
    ExperimentConfig,
    KDET,
    LogEvidenceUpdater,
    SignFlipGroup,
    build_default_sign_flip_strategies,
    run_experiment,
)


def test_sign_flip_group_action_and_orbit() -> None:
    space = EuclideanSpace(dim=2)
    group = SignFlipGroup(space=space)
    x = np.array([1.5, -2.0])

    assert list(group.elements()) == [1, -1]
    np.testing.assert_allclose(group.action(1, x), x)
    np.testing.assert_allclose(group.action(-1, x), -x)


def test_dynamic_t_evidence_is_orbit_normalized() -> None:
    space = EuclideanSpace(dim=1)
    group = SignFlipGroup(space=space)
    updater = LogEvidenceUpdater.from_dynamic_T(
        lambda past: KDET(past, bandwidth=0.75, floor=1e-9)
    )
    updater.observations.append(np.array([0.4]))

    x = np.array([1.2])
    rng = np.random.default_rng(0)
    orbit_log_evidence = [
        updater.get_log_evidence(group, group.action(g, x), rng)
        for g in group.elements()
    ]

    orbit_mean = float(np.mean(np.exp(orbit_log_evidence)))
    assert orbit_mean == pytest.approx(1.0, rel=1e-12, abs=1e-12)
    assert all(math.isfinite(value) for value in orbit_log_evidence)


def test_non_neural_experiment_is_reproducible() -> None:
    config = ExperimentConfig(runs=8, step=5, seed=123, include_neural=False)

    first = run_experiment(config, build_default_sign_flip_strategies(config))
    second = run_experiment(config, build_default_sign_flip_strategies(config))

    first_metrics = [
        (
            result.strategy_name,
            result.type1_error,
            result.type2_error,
            result.power,
            result.avg_null_stopping_time,
            result.avg_alt_stopping_time,
        )
        for result in first.results
    ]
    second_metrics = [
        (
            result.strategy_name,
            result.type1_error,
            result.type2_error,
            result.power,
            result.avg_null_stopping_time,
            result.avg_alt_stopping_time,
        )
        for result in second.results
    ]

    assert first_metrics == second_metrics
