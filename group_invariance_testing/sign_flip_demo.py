from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from .experiment_driver import ExperimentConfig, ExperimentResult, build_default_sign_flip_strategies, default_log_path as experiment_log_path, run_experiment
except ImportError:
    from experiment_driver import ExperimentConfig, ExperimentResult, build_default_sign_flip_strategies, default_log_path as experiment_log_path, run_experiment


def default_log_path() -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return experiment_log_path(f"{timestamp}-sign_flip_experiment.log")


def print_strategy_result(result: ExperimentResult) -> None:
    print(f"Finished {result.strategy_name}: type I error={result.type1_error:.6g}, type II error={result.type2_error:.6g}, power={result.power:.6g}, avg null stop={result.avg_null_stopping_time:.6g}, avg alternative stop={result.avg_alt_stopping_time:.6g}, runtime={result.runtime_seconds:.6g}s", flush=True)


def make_neural_step_printer(every: int = 10):
    neural_steps = 0
    def print_neural_step(event: dict[str, object]) -> None:
        nonlocal neural_steps
        strategy_name = getattr(event["strategy"], "name", "")
        if "Neural" not in strategy_name:
            return
        neural_steps += 1
        if neural_steps % every == 0:
            print(f"{strategy_name}: processed {neural_steps} neural data steps (phase={event['phase']}, trial={event['trial_index']}, step={event['step_index']}, log evidence={float(event['log_evidence']):.6g})", flush=True)
    return print_neural_step


def main() -> None:
    config = ExperimentConfig(runs=1000, step=50, include_neural=True)
    strategies = build_default_sign_flip_strategies(config)
    print(f"Running sign-flip demo with {config.runs} runs per distribution, {config.step} max steps per run, and strategies: {', '.join(strategy.name for strategy in strategies)}", flush=True)
    summary = run_experiment(config=config, strategies=strategies, log_path=default_log_path(), strategy_result_callback=print_strategy_result, step_callback=make_neural_step_printer(every=10))
    if summary.log_path is not None:
        print(f"Log written to: {summary.log_path}")


if __name__ == "__main__":
    main()
