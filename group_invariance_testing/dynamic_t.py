from __future__ import annotations

import math

import numpy as np

try:
    from .core import DynamicT
except ImportError:
    from core import DynamicT

_TINY_POSITIVE = float(np.finfo(float).tiny)


def _as_vector(x) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1:
        raise ValueError("DynamicT methods expect 1D vectors in R^n.")
    return vector


def _stack_past_data(past_data: list[np.ndarray]) -> np.ndarray | None:
    if len(past_data) == 0:
        return None
    vectors = [_as_vector(x) for x in past_data]
    dim = vectors[0].shape[0]
    if any(vector.shape != (dim,) for vector in vectors):
        raise ValueError("All past observations must have the same dimension.")
    return np.stack(vectors, axis=0)


class MovingGaussianT(DynamicT):
    """Diagonal Gaussian score fit from the empirical mean and variance of past data."""

    def __init__(self, past_data: list[np.ndarray], variance_floor: float = 1e-6, prior_variance: float = 1.0):
        self.variance_floor = float(variance_floor)
        self.prior_variance = float(prior_variance)
        if self.variance_floor <= 0:
            raise ValueError("variance_floor must be positive.")
        if self.prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")
        self.samples = _stack_past_data(past_data)
        if self.samples is None:
            self.mean = None
            self.variance = None
        else:
            self.mean = np.mean(self.samples, axis=0)
            if self.samples.shape[0] <= 1:
                empirical_variance = np.full(self.samples.shape[1], self.prior_variance, dtype=float)
            else:
                empirical_variance = np.var(self.samples, axis=0, ddof=1)
            self.variance = np.maximum(empirical_variance, self.variance_floor)

    def __call__(self, x) -> float:
        vector = _as_vector(x)
        if self.mean is None:
            mean = np.zeros_like(vector)
            variance = np.full(vector.shape, self.prior_variance, dtype=float)
        else:
            if vector.shape != self.mean.shape:
                raise ValueError(f"x must have shape {self.mean.shape}.")
            mean = self.mean
            variance = self.variance
        sq_term = np.sum(((vector - mean) ** 2) / variance)
        log_norm = np.sum(np.log(2.0 * math.pi * variance))
        return float(max(math.exp(-0.5 * (log_norm + sq_term)), _TINY_POSITIVE))


class MeanOnlyGaussianT(DynamicT):
    """Diagonal Gaussian score with empirical mean and fixed known variance."""

    def __init__(self, past_data: list[np.ndarray], fixed_variance: float = 1.0):
        self.fixed_variance = float(fixed_variance)
        if self.fixed_variance <= 0:
            raise ValueError("fixed_variance must be positive.")
        self.samples = _stack_past_data(past_data)
        self.mean = None if self.samples is None else np.mean(self.samples, axis=0)

    def __call__(self, x) -> float:
        vector = _as_vector(x)
        if self.mean is None:
            mean = np.zeros_like(vector)
        else:
            if vector.shape != self.mean.shape:
                raise ValueError(f"x must have shape {self.mean.shape}.")
            mean = self.mean
        sq_term = np.sum((vector - mean) ** 2) / self.fixed_variance
        log_norm = vector.shape[0] * math.log(2.0 * math.pi * self.fixed_variance)
        return float(max(math.exp(-0.5 * (log_norm + sq_term)), _TINY_POSITIVE))


class KDET(DynamicT):
    """Gaussian-kernel KDE on the past observations with an optional positive floor."""

    def __init__(self, past_data: list[np.ndarray], bandwidth: float = 1.0, floor: float = 0.0):
        self.bandwidth = float(bandwidth)
        self.floor = float(floor)
        if self.bandwidth <= 0:
            raise ValueError("bandwidth must be positive.")
        if self.floor < 0:
            raise ValueError("floor must be non-negative.")
        self.samples = _stack_past_data(past_data)
        self.dim = None if self.samples is None else self.samples.shape[1]
        self.normalization = 1.0 if self.dim is None else 1.0 / ((math.sqrt(2.0 * math.pi) * self.bandwidth) ** self.dim)

    def __call__(self, x) -> float:
        if self.samples is None:
            return float(max(1.0 + self.floor, _TINY_POSITIVE))
        vector = _as_vector(x)
        if vector.shape != (self.dim,):
            raise ValueError(f"x must have shape ({self.dim},).")
        diffs = self.samples - vector
        sq_norms = np.sum(diffs * diffs, axis=1)
        kernel_values = np.exp(-sq_norms / (2.0 * self.bandwidth * self.bandwidth))
        density = self.normalization * float(np.mean(kernel_values))
        return float(max(self.floor + density, _TINY_POSITIVE))


RunningMeanGaussianT = MovingGaussianT
KDE_T = KDET

__all__ = ["KDET", "KDE_T", "MeanOnlyGaussianT", "MovingGaussianT", "RunningMeanGaussianT"]
