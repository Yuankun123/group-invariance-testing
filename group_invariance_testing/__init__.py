"""Top-level exports for the group-invariance testing framework."""

from .DynamicT import KDET, KDE_T, MeanOnlyGaussianT, MovingGaussianT, RunningMeanGaussianT
from .core import ActionGroup, DataGenerator, DynamicT, LogEvidenceUpdater, SampleSpace, SequentialTest
from .experiment_driver import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentSummary,
    TStrategySpec,
    build_default_sign_flip_strategies,
    run_experiment,
)
from .neuralT import (
    GeneralOrbitNeuralLogEvidenceUpdater,
    GeneralOrbitNeuralT,
    SignFlipNeuralLogEvidenceUpdater,
    SignFlipNeuralT,
)
from .orcaleT import OracleT
from .space_and_group import EuclideanSpace, NormalDistributionOnRn, SignFlipGroup

__all__ = [
    "ActionGroup",
    "DataGenerator",
    "DynamicT",
    "EuclideanSpace",
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSummary",
    "GeneralOrbitNeuralLogEvidenceUpdater",
    "GeneralOrbitNeuralT",
    "KDET",
    "KDE_T",
    "LogEvidenceUpdater",
    "MeanOnlyGaussianT",
    "MovingGaussianT",
    "NormalDistributionOnRn",
    "OracleT",
    "RunningMeanGaussianT",
    "SampleSpace",
    "SequentialTest",
    "SignFlipGroup",
    "SignFlipNeuralLogEvidenceUpdater",
    "SignFlipNeuralT",
    "TStrategySpec",
    "build_default_sign_flip_strategies",
    "run_experiment",
]
