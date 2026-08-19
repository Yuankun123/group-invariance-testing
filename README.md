# Group Invariance Testing

A research implementation of anytime-valid sequential tests for group invariance, with concrete kernel-, Gaussian-, and neural-network-based testing statistics.

## Scope and attribution

The **anytime-valid group-invariance testing framework itself is due to Nick W. Koning**. The framework and its theoretical guarantees are developed in:

> Nick W. Koning, *Post-hoc and Anytime Valid Permutation and Group Invariance Testing*, arXiv:2310.01153v3.  
> https://arxiv.org/abs/2310.01153v3

Koning's framework allows flexible positive test statistics / score functions to be plugged into orbit-normalized evidence processes. **Our work in this project is to design, implement, and evaluate three concrete choices of testing statistic within that framework:**

1. **Kernel-density based statistics** using Gaussian-kernel KDE on past observations.
2. **Adaptive Gaussian statistics** using online estimates of distributional parameters.
3. **Neural-network based statistics**, trained online to learn a score that separates observations from their group-transformed counterparts. This is the main method explored in the project, especially for settings where simple parametric scores are poorly matched to the data distribution.

An oracle Gaussian location-shift statistic is also included as a benchmark; it is not presented as one of our proposed adaptive methods.

## Repository provenance

This repository is a **public cleanup snapshot**, not the original development repository. The project was developed collaboratively in a separate private repository and later curated here for public release.

Consequently, **the Git history of this repository does not represent the actual development process or individual contribution history**. Internal meeting notes, raw experiment logs, deprecated prototypes, extracted reference-paper text, local copies of third-party papers, and the original private Git history are intentionally excluded.

## What is implemented

- Generic interfaces for sample spaces, group actions, data generators, and sequential log-evidence updates.
- Sign-flip invariance on Euclidean data.
- Gaussian-kernel density scores (`KDET`).
- Adaptive Gaussian scores using empirical means and variances.
- Stateful neural log-evidence updaters, including a specialized sign-flip construction and a general finite-group orbit construction.
- An oracle Gaussian location-shift benchmark.
- Simulation drivers for comparing Type-I error, power, stopping time, and runtime.
- A strongly non-normal sign-flip experiment based on a signed lognormal mixture.

## Representative result

One representative development experiment used a strongly non-normal sign-flip problem. The null distribution was a symmetric signed two-component lognormal mixture, while under the alternative the magnitude distribution was unchanged but the signs were imbalanced: 70% positive and 30% negative. The experiment used 200 trials per distribution, a maximum of 60 sequential observations per trial, and significance level `alpha = 0.05`.

| Statistic | Type-I error | Power | Avg. alternative stopping time |
|---|---:|---:|---:|
| KDE | 0.010 | 0.205 | 51.11 |
| Adaptive Gaussian | 0.010 | 0.220 | 51.76 |
| **Neural** | **0.025** | **0.395** | **49.83** |

In this recorded setting, the neural statistic achieved substantially higher empirical power than the KDE and adaptive-Gaussian alternatives while the observed Type-I error remained below the nominal 0.05 level. These numbers are included as an illustrative result from the project rather than as a comprehensive benchmark.

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

## Testing

The lightweight test suite does not require PyTorch; neural tests are skipped when PyTorch is unavailable.

```bash
pip install -e ".[test]"
pytest -q
```

GitHub Actions runs this suite on pushes and pull requests.

## Repository layout

```text
group_invariance_testing/   Core implementation and experiment drivers
tests/                      Lightweight correctness and reproducibility checks
.github/workflows/          Continuous-integration configuration
README.md                   Project overview and usage
pyproject.toml              Package and dependency metadata
```

## Contributors

- **Yuankun (Kunko) Zou** — lead developer; designed and implemented the software framework, core testing methods, and experiment pipeline.
- **Jiajun (William) Du** — contributed code optimization and implementation improvements.
- **Johnny Jiang** — contributed high-dimensional experiments and evaluation.

## Related work

- Nick W. Koning, *Post-hoc and Anytime Valid Permutation and Group Invariance Testing*, arXiv:2310.01153v3.
- Nick W. Koning and Jesse Hemerik, *More Efficient Exact Group-Invariance Testing: using a Representative Subgroup*, *Biometrika* 111(2), 2024.

## Status

This is a curated research-code snapshot intended to make the project's implementation and main experimental ideas easy to inspect and reproduce. The original development workspace remains private.
