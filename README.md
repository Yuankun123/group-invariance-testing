# Group Invariance Testing

A cleaned public implementation of anytime-valid sequential tests for group invariance. The code explores adaptive positive scoring rules whose evidence is normalized over group orbits, with kernel, Gaussian, oracle, and neural scoring strategies.

The statistical framework is based on Nick W. Koning, *Post-hoc and Anytime Valid Permutation and Group Invariance Testing* (arXiv:2310.01153).

## What is implemented

- A generic interface for sample spaces, group actions, data generators, and sequential log-evidence updates.
- Sign-flip invariance on Euclidean data.
- Gaussian-kernel density scores (`KDET`).
- Adaptive Gaussian scores using empirical means and variances.
- An oracle Gaussian location-shift benchmark.
- Stateful neural log-evidence updaters, including a specialized sign-flip construction and a general finite-group orbit construction.
- Simulation drivers for comparing Type-I error, power, stopping time, and runtime.
- A strongly non-normal sign-flip experiment based on a signed lognormal mixture.

## Installation

Python 3.10+ is recommended.

```bash
pip install -e .
```

For the neural strategies, install the optional PyTorch dependency:

```bash
pip install -e ".[neural]"
```

## Quick start

A lightweight non-neural experiment can be run from Python:

```python
from group_invariance_testing import (
    ExperimentConfig,
    build_default_sign_flip_strategies,
    run_experiment,
)

config = ExperimentConfig(runs=100, step=50, include_neural=False)
strategies = build_default_sign_flip_strategies(config)
summary = run_experiment(config, strategies)

for result in summary.results:
    print(result)
```

The included demos can also be run as modules:

```bash
python -m group_invariance_testing.sign_flip_demo
python -m group_invariance_testing.non_normal_sign_flip_demo --runs 100 --steps 50
```

The first demo enables the neural strategy by default and can therefore be substantially slower.

## Repository layout

```text
group_invariance_testing/   Core implementation and experiment drivers
README.md                   Project overview and usage
pyproject.toml              Minimal package/dependency metadata
```

This public repository intentionally excludes internal meeting notes, raw experiment logs, deprecated prototypes, extracted reference-paper text, and local copies of third-party papers from the original research workspace.

## Related work

- Nick W. Koning, *Post-hoc and Anytime Valid Permutation and Group Invariance Testing*, arXiv:2310.01153.
- Nick W. Koning and Jesse Hemerik, *More Efficient Exact Group-Invariance Testing: using a Representative Subgroup*, Biometrika 111(2), 2024.

## Status

This is research code accompanying a completed collaborative project. The public repository is intended to provide a concise, reproducible snapshot of the implementation; the API may still be cleaned further without changing the underlying experiments.
