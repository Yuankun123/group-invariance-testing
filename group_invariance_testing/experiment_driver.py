from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import math
from pathlib import Path
import time
from typing import Callable
import warnings

import numpy as np

try:
    from .core import LogEvidenceUpdater
    from .dynamic_t import KDET, MeanOnlyGaussianT, MovingGaussianT
    from .oracle_t import OracleT
    from .space_and_group import EuclideanSpace, NormalDistributionOnRn, SignFlipGroup
except ImportError:
    from core import LogEvidenceUpdater
    from dynamic_t import KDET, MeanOnlyGaussianT, MovingGaussianT
    from oracle_t import OracleT
    from space_and_group import EuclideanSpace, NormalDistributionOnRn, SignFlipGroup

try:
    from .neural_t import (
        SignFlipNeuralLogEvidenceUpdater,
        TORCH_AVAILABLE as _NEURAL_TORCH_AVAILABLE,
        TORCH_IMPORT_ERROR as _NEURAL_IMPORT_ERROR,
    )
except ImportError:
    try:
        from neural_t import (
            SignFlipNeuralLogEvidenceUpdater,
            TORCH_AVAILABLE as _NEURAL_TORCH_AVAILABLE,
            TORCH_IMPORT_ERROR as _NEURAL_IMPORT_ERROR,
        )
    except ImportError as exc:
        SignFlipNeuralLogEvidenceUpdater = None
        _NEURAL_TORCH_AVAILABLE = False
        _NEURAL_IMPORT_ERROR = exc


@dataclass(frozen=True)
class ExperimentConfig:
    runs: int = 10000
    alpha: float = 0.05
    step: int = 50
    dim: int = 1
    sigma: float = 1.0
    alt_shift: float = 1.0
    seed: int = 0
    kdet_floor: float = 0.0
    moving_variance_floor: float = 1e-6
    oracle_shift: float | None = None
    include_neural: bool = False
    neural_hidden_width: int = 32
    neural_hidden_depth: int = 1
    neural_activation: str = "tanh"
    neural_min_train_samples: int = 10
    neural_update_epochs: int = 5
    neural_replay_batch_size: int = 32
    neural_lr: float = 1e-2
    neural_weight_decay: float = 1e-4
    neural_clip_value: float = 20.0
    neural_epochs: int | None = None

    @property
    def null_mean(self) -> np.ndarray:
        return np.zeros(self.dim, dtype=float)

    @property
    def alt_mean(self) -> np.ndarray:
        mean = np.zeros(self.dim, dtype=float)
        mean[0] = self.alt_shift
        return mean

    @property
    def effective_oracle_shift(self) -> float:
        return self.alt_shift if self.oracle_shift is None else float(self.oracle_shift)

    @property
    def effective_neural_update_epochs(self) -> int:
        if self.neural_epochs is not None:
            return int(self.neural_epochs)
        return int(self.neural_update_epochs)


@dataclass(frozen=True)
class TStrategySpec:
    name: str
    description: str
    factory: object
    factory_kind: str = "dynamic_t"


@dataclass(frozen=True)
class ExperimentResult:
    strategy_name: str
    description: str
    runtime_seconds: float
    type1_error: float
    type2_error: float
    power: float
    avg_null_stopping_time: float
    avg_alt_stopping_time: float


@dataclass(frozen=True)
class ExperimentSummary:
    config: ExperimentConfig
    results: list[ExperimentResult]
    log_text: str
    log_path: Path | None


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_log_dir() -> Path:
    return repo_root() / "log"


def default_log_path(file_name: str) -> Path:
    return default_log_dir() / file_name


def _warn_neural_strategy_skipped(reason: str) -> None:
    warnings.warn(
        (
            "include_neural=True was requested, but the neural strategy was skipped. "
            f"Reason: {reason}"
        ),
        RuntimeWarning,
        stacklevel=3,
    )


def build_default_sign_flip_strategies(config: ExperimentConfig) -> list[TStrategySpec]:
    strategies = [
        TStrategySpec(
            "KDET",
            f"Gaussian-kernel KDE on past data with floor={config.kdet_floor:g}.",
            partial(KDET, floor=config.kdet_floor),
        ),
        TStrategySpec(
            "OracleT",
            (
                "Oracle Gaussian location-shift score "
                f"(shift={config.effective_oracle_shift:g}, sigma={config.sigma:g})."
            ),
            partial(
                OracleT,
                shift=config.effective_oracle_shift,
                sigma=config.sigma,
                coordinate=0,
            ),
        ),
        TStrategySpec(
            "MovingGaussianT",
            (
                "Diagonal Gaussian score fit from empirical mean/variance "
                f"(variance_floor={config.moving_variance_floor:g})."
            ),
            partial(
                MovingGaussianT,
                variance_floor=config.moving_variance_floor,
                prior_variance=config.sigma * config.sigma,
            ),
        ),
        TStrategySpec(
            "MeanOnlyGaussianT",
            (
                "Diagonal Gaussian score with empirical mean and fixed variance="
                f"{config.sigma * config.sigma:g}."
            ),
            partial(
                MeanOnlyGaussianT,
                fixed_variance=config.sigma * config.sigma,
            ),
        ),
    ]

    if config.include_neural:
        if SignFlipNeuralLogEvidenceUpdater is None or not _NEURAL_TORCH_AVAILABLE:
            reason = (
                str(_NEURAL_IMPORT_ERROR)
                if _NEURAL_IMPORT_ERROR is not None
                else "PyTorch is unavailable."
            )
            _warn_neural_strategy_skipped(reason)
        else:
            strategies.append(
                TStrategySpec(
                    "SignFlipNeuralLogEvidenceUpdater",
                    (
                        "Stateful sign-flip neural log-evidence updater trained online "
                        "after each scored observation."
                    ),
                    partial(
                        SignFlipNeuralLogEvidenceUpdater,
                        hidden_width=config.neural_hidden_width,
                        hidden_depth=config.neural_hidden_depth,
                        activation=config.neural_activation,
                        min_train_samples=config.neural_min_train_samples,
                        update_epochs=config.effective_neural_update_epochs,
                        replay_batch_size=config.neural_replay_batch_size,
                        lr=config.neural_lr,
                        weight_decay=config.neural_weight_decay,
                        clip_value=config.neural_clip_value,
                        seed=config.seed,
                    ),
                    "log_evidence_updater",
                )
            )
    return strategies


def _make_seed_stream(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    max_seed = np.iinfo(np.uint32).max
    return (
        rng.integers(0, max_seed, size=config.runs, dtype=np.uint32),
        rng.integers(0, max_seed, size=config.runs, dtype=np.uint32),
    )


def run_trial(
    mu: np.ndarray,
    seed: int,
    config: ExperimentConfig,
    strategy: TStrategySpec,
    phase: str = "",
    trial_index: int | None = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, float | int | bool | str]:
    rng = np.random.default_rng(seed)
    space = EuclideanSpace(dim=mu.shape[0])
    group = SignFlipGroup(space=space)
    data = NormalDistributionOnRn(space=space, sigma=config.sigma, mu=mu)

    if strategy.factory_kind == "dynamic_t":
        log_evi = LogEvidenceUpdater.from_dynamic_T(strategy.factory)
    elif strategy.factory_kind == "log_evidence_updater":
        log_evi = strategy.factory(input_dim=mu.shape[0])
        if not isinstance(log_evi, LogEvidenceUpdater):
            raise TypeError("log_evidence_updater factories must return a LogEvidenceUpdater.")
    else:
        raise ValueError(f"Unknown strategy factory_kind: {strategy.factory_kind!r}.")

    threshold = math.log(1.0 / config.alpha)
    rejected = False
    for step_index in range(1, config.step + 1):
        new_obs = data.get_random_elem(rng)
        log_evi.one_step(group, new_obs, rng)
        rejected = log_evi.current_log_evidence >= threshold
        if step_callback is not None:
            step_callback(
                {
                    "strategy": strategy,
                    "phase": phase,
                    "trial_index": trial_index,
                    "step_index": step_index,
                    "num_observations": len(log_evi.observations),
                    "log_evidence": log_evi.current_log_evidence,
                    "rejected": rejected,
                }
            )
        if rejected:
            break

    return {
        "rejected": rejected,
        "num_observations": len(log_evi.observations),
        "log_evidence": log_evi.current_log_evidence,
        "strategy_name": strategy.name,
    }


def _evaluate_strategy(
    config: ExperimentConfig,
    strategy: TStrategySpec,
    null_seeds: np.ndarray,
    alt_seeds: np.ndarray,
    step_callback: Callable[[dict[str, object]], None] | None = None,
) -> ExperimentResult:
    started_at = time.perf_counter()
    null_results = [
        run_trial(
            config.null_mean,
            int(seed),
            config,
            strategy,
            "null",
            i,
            step_callback,
        )
        for i, seed in enumerate(null_seeds, 1)
    ]
    alt_results = [
        run_trial(
            config.alt_mean,
            int(seed),
            config,
            strategy,
            "alternative",
            i,
            step_callback,
        )
        for i, seed in enumerate(alt_seeds, 1)
    ]
    type1_error = float(np.mean([r["rejected"] for r in null_results]))
    type2_error = float(np.mean([not r["rejected"] for r in alt_results]))
    return ExperimentResult(
        strategy.name,
        strategy.description,
        time.perf_counter() - started_at,
        type1_error,
        type2_error,
        1.0 - type2_error,
        float(np.mean([r["num_observations"] for r in null_results])),
        float(np.mean([r["num_observations"] for r in alt_results])),
    )


def render_experiment_log(
    config: ExperimentConfig,
    results: list[ExperimentResult],
) -> str:
    lines = [
        "Sign-flip experiment log",
        f"Runs: {config.runs}",
        f"Alpha: {config.alpha:g}",
        f"Steps per run: {config.step}",
        f"Dimension: {config.dim}",
        f"Seed: {config.seed}",
        "",
    ]
    for result in results:
        lines.extend(
            [
                f"T strategy: {result.strategy_name}",
                f"Description: {result.description}",
                f"Runtime (seconds): {result.runtime_seconds:.6g}",
                f"Type I error: {result.type1_error:.6g}",
                f"Type II error: {result.type2_error:.6g}",
                f"Power: {result.power:.6g}",
                f"Average stop step under null: {result.avg_null_stopping_time:.6g}",
                f"Average stop step under alternative: {result.avg_alt_stopping_time:.6g}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def run_experiment(
    config: ExperimentConfig,
    strategies: list[TStrategySpec],
    log_path: str | Path | None = None,
    strategy_result_callback: Callable[[ExperimentResult], None] | None = None,
    step_callback: Callable[[dict[str, object]], None] | None = None,
) -> ExperimentSummary:
    null_seeds, alt_seeds = _make_seed_stream(config)
    results = []
    for strategy in strategies:
        result = _evaluate_strategy(
            config,
            strategy,
            null_seeds,
            alt_seeds,
            step_callback,
        )
        results.append(result)
        if strategy_result_callback is not None:
            strategy_result_callback(result)

    log_text = render_experiment_log(config, results)
    resolved_log_path = None if log_path is None else Path(log_path)
    if resolved_log_path is not None:
        resolved_log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_log_path.write_text(log_text, encoding="utf-8")

    return ExperimentSummary(config, results, log_text, resolved_log_path)


__all__ = [
    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentSummary",
    "TStrategySpec",
    "build_default_sign_flip_strategies",
    "default_log_dir",
    "default_log_path",
    "repo_root",
    "render_experiment_log",
    "run_experiment",
    "run_trial",
]
