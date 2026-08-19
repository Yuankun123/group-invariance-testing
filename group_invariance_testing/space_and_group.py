from __future__ import annotations

from typing import Callable, Generator
import numpy as np

try:
    from .core import ActionGroup, DataGenerator, SampleSpace
except ImportError:
    from core import ActionGroup, DataGenerator, SampleSpace


class EuclideanSpace(SampleSpace[np.ndarray]):
    def __init__(self, dim: int):
        dim = int(dim)
        if dim <= 0:
            raise ValueError("dim must be positive.")
        super().__init__(np.ndarray)
        self.dim = dim

    def is_element(self, x) -> bool:
        vector = np.asarray(x, dtype=float)
        return vector.ndim == 1 and vector.shape == (self.dim,)


class SignFlipGroup(ActionGroup[int, np.ndarray]):
    """The two-element sign-flip group {+1, -1} acting on R^n."""

    def __init__(self, space: EuclideanSpace | None = None):
        self.space = space

    def action(self, g: int, x: np.ndarray) -> np.ndarray:
        if g not in (-1, 1):
            raise ValueError("sign-flip elements must be +1 or -1.")
        vector = np.asarray(x, dtype=float)
        if self.space is not None and not self.space.is_element(vector):
            raise ValueError(f"x must belong to R^{self.space.dim}.")
        return int(g) * vector

    def elements(self) -> Generator[int, None, None]:
        yield 1
        yield -1

    def random_elem(self, rng) -> int:
        if rng is None:
            rng = np.random.default_rng()
        return int(rng.choice(np.array([1, -1], dtype=int)))

    def orbit_average(self, f: Callable[[int], float]) -> float:
        values = [f(g) for g in self.elements()]
        return float(sum(values) / len(values))


class NormalDistributionOnRn(DataGenerator[np.ndarray]):
    def __init__(self, space: EuclideanSpace, sigma: float, mu: np.ndarray):
        super().__init__(space)
        self.sigma = float(sigma)
        self.mu = np.asarray(mu, dtype=float)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive.")
        if not self.space.is_element(self.mu):
            raise ValueError(f"mu must have shape ({self.space.dim},).")

    def get_random_elem(self, rng) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.normal(loc=self.mu, scale=self.sigma, size=self.space.dim)


__all__ = ["EuclideanSpace", "NormalDistributionOnRn", "SignFlipGroup"]
