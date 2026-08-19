import numpy as np
import pytest

pytest.importorskip("torch")

from group_invariance_testing import (
    EuclideanSpace,
    SignFlipGroup,
    SignFlipNeuralLogEvidenceUpdater,
)


def test_sign_flip_neural_evidence_is_orbit_normalized() -> None:
    group = SignFlipGroup(space=EuclideanSpace(dim=1))
    updater = SignFlipNeuralLogEvidenceUpdater(
        input_dim=1,
        min_train_samples=0,
        update_epochs=0,
        clip_value=3.0,
        seed=0,
    )
    rng = np.random.default_rng(0)
    x = np.array([1.25])

    logs = [
        updater.get_log_evidence(group, group.action(g, x), rng)
        for g in group.elements()
    ]
    orbit_mean = float(np.mean(np.exp(logs)))

    assert orbit_mean == pytest.approx(1.0, rel=1e-6, abs=1e-6)
