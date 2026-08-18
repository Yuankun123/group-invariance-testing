from __future__ import annotations

import argparse
from dataclasses import replace
import math
from pathlib import Path
import time
import numpy as np

try:
    from .core import LogEvidenceUpdater
    from .experiment_driver import ExperimentConfig, ExperimentResult, TStrategySpec, build_default_sign_flip_strategies, default_log_path
    from .space_and_group import EuclideanSpace, SignFlipGroup
except ImportError:
    from core import LogEvidenceUpdater
    from experiment_driver import ExperimentConfig, ExperimentResult, TStrategySpec, build_default_sign_flip_strategies, default_log_path
    from space_and_group import EuclideanSpace, SignFlipGroup

NULL_DESCRIPTION = "A symmetric signed two-component lognormal mixture: draw a positive magnitude from 0.85 LogNormal(0, 0.65^2) + 0.15 LogNormal(2.5, 0.35^2), then multiply it by an independent uniform sign."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sign-flip tests on strongly non-normal distributions.")
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--positive-percentage", type=float, default=60.0)
    parser.add_argument("--log-path", type=Path, default=default_log_path("non_normal_sign_flip_experiment.log"))
    args = parser.parse_args()
    if not 0.0 <= args.positive_percentage <= 100.0:
        parser.error("--positive-percentage must be between 0 and 100.")
    return args


def sample_lognormal_mixture(rng: np.random.Generator) -> float:
    return float(rng.lognormal(mean=0.0, sigma=0.65)) if rng.random() < 0.85 else float(rng.lognormal(mean=2.5, sigma=0.35))


def sample_observation(rng, invariant, alt_positive_probability):
    magnitude = sample_lognormal_mixture(rng)
    positive_probability = 0.5 if invariant else alt_positive_probability
    sign = 1 if rng.random() < positive_probability else -1
    return np.array([sign * magnitude], dtype=float)


def build_strategies(config):
    wanted = {"KDET", "MovingGaussianT", "SignFlipNeuralLogEvidenceUpdater"}
    return [s for s in build_default_sign_flip_strategies(config) if s.name in wanted]


def run_trial(invariant, seed, config, strategy, alt_positive_probability):
    rng = np.random.default_rng(seed)
    group = SignFlipGroup(space=EuclideanSpace(dim=1))
    log_evidence = LogEvidenceUpdater.from_dynamic_T(strategy.factory) if strategy.factory_kind == "dynamic_t" else strategy.factory(input_dim=1)
    threshold = math.log(1.0 / config.alpha)
    for _ in range(config.step):
        observation = sample_observation(rng, invariant, alt_positive_probability)
        log_evidence.one_step(group, observation, rng)
        if log_evidence.current_log_evidence >= threshold:
            return True, len(log_evidence.observations)
    return False, len(log_evidence.observations)


def evaluate_strategy(config, strategy, null_seeds, alt_seeds, alt_positive_probability):
    started_at = time.perf_counter()
    null_results = [run_trial(True, int(seed), config, strategy, alt_positive_probability) for seed in null_seeds]
    alt_results = [run_trial(False, int(seed), config, strategy, alt_positive_probability) for seed in alt_seeds]
    type1_error = float(np.mean([r for r, _ in null_results]))
    type2_error = float(np.mean([not r for r, _ in alt_results]))
    return ExperimentResult(strategy.name, strategy.description, time.perf_counter() - started_at, type1_error, type2_error, 1.0 - type2_error, float(np.mean([s for _, s in null_results])), float(np.mean([s for _, s in alt_results])))


def run(config, log_path: Path, alt_positive_probability: float = 0.60):
    seed_rng = np.random.default_rng(config.seed)
    max_seed = np.iinfo(np.uint32).max
    null_seeds = seed_rng.integers(0, max_seed, size=config.runs, dtype=np.uint32)
    alt_seeds = seed_rng.integers(0, max_seed, size=config.runs, dtype=np.uint32)
    results = []
    for strategy in build_strategies(config):
        result = evaluate_strategy(config, strategy, null_seeds, alt_seeds, alt_positive_probability)
        results.append(result)
        print(f"Finished {result.strategy_name}: type I error={result.type1_error:.4f}, power={result.power:.4f}, runtime={result.runtime_seconds:.2f}s", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Strongly non-normal sign-flip experiment log", f"Runs per distribution: {config.runs}", f"Alpha: {config.alpha:g}", f"Maximum steps per run: {config.step}", f"Invariant null distribution: {NULL_DESCRIPTION}", ""]
    for result in results:
        lines.extend([f"T strategy: {result.strategy_name}", f"Type I error: {result.type1_error:.6g}", f"Power: {result.power:.6g}", ""])
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Log written to: {log_path}")
    return results


def main() -> None:
    args = parse_args()
    config = replace(ExperimentConfig(include_neural=True), runs=args.runs, step=args.steps, seed=args.seed)
    run(config, args.log_path, args.positive_percentage / 100.0)


if __name__ == "__main__":
    main()
