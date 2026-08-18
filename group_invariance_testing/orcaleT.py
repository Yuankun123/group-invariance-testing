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
        raise ValueError("OracleT expects 1D vectors in R^n.")
    return vector


class OracleT(DynamicT):
    """Oracle Gaussian location-shift score with the alternative baked in."""

    def __init__(self, past_data: list[np.ndarray], shift: float = 1.0, sigma: float = 1.0, coordinate: int = 0):
        del past_data
        self.shift = float(shift)
        self.sigma = float(sigma)
        self.coordinate = int(coordinate)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if self.coordinate < 0:
            raise ValueError("coordinate must be non-negative.")
        self.kappa = self.shift / (self.sigma * self.sigma)

    def __call__(self, x) -> float:
        vector = _as_vector(x)
        if self.coordinate >= vector.shape[0]:
            raise ValueError(f"x must have at least {self.coordinate + 1} coordinates.")
        return float(max(math.exp(self.kappa * vector[self.coordinate]), _TINY_POSITIVE))


__all__ = ["OracleT"]
