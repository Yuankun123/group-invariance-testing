from __future__ import annotations

import math
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:
    torch = None
    F = None
    _TORCH_IMPORT_ERROR: ImportError | None = exc
else:
    _TORCH_IMPORT_ERROR = None

try:
    from .core import DynamicT, LogEvidenceUpdater
    from .neural_models import build_scalar_mlp
except ImportError:
    from core import DynamicT, LogEvidenceUpdater
    from neural_models import build_scalar_mlp

_PYTORCH_REQUIRED_MESSAGE = "Neural strategies require PyTorch. Install torch or disable neural strategies."


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(_PYTORCH_REQUIRED_MESSAGE) from _TORCH_IMPORT_ERROR


def _as_vector(x) -> np.ndarray:
    vector = np.asarray(x, dtype=float)
    if vector.ndim != 1:
        raise ValueError("Neural methods expect 1D vectors in R^n.")
    return vector


def _stack_past_data(past_data: list[np.ndarray]) -> np.ndarray | None:
    if len(past_data) == 0:
        return None
    vectors = [_as_vector(x) for x in past_data]
    dim = vectors[0].shape[0]
    if any(vector.shape != (dim,) for vector in vectors):
        raise ValueError("All past observations must have the same dimension.")
    return np.stack(vectors, axis=0)


def _build_model(input_dim: int, model_factory, hidden_width: int, hidden_depth: int, activation: str):
    if model_factory is not None:
        return model_factory(input_dim)
    return build_scalar_mlp(input_dim=input_dim, hidden_width=hidden_width, hidden_depth=hidden_depth, activation=activation)


def _model_scores(model, vectors: torch.Tensor) -> torch.Tensor:
    scores = model(vectors)
    if scores.ndim == 2 and scores.shape[1] == 1:
        scores = scores.squeeze(1)
    if scores.ndim != 1:
        raise ValueError("Neural score models must return shape (batch,) or (batch, 1).")
    return scores


class _ReplayMixin:
    def _sample_replay_batch(self, rng) -> np.ndarray:
        observations = [_as_vector(x) for x in self.observations]
        if len(observations) == 0:
            raise ValueError("Cannot sample replay batch without observations.")
        if len(observations) <= self.replay_batch_size:
            return np.stack(observations, axis=0)
        if rng is None:
            rng = np.random.default_rng()
        newest = observations[-1]
        if self.replay_batch_size == 1:
            return newest.reshape(1, -1)
        indices = rng.choice(len(observations) - 1, size=self.replay_batch_size - 1, replace=False)
        batch = [observations[int(index)] for index in indices]
        batch.append(newest)
        return np.stack(batch, axis=0)


class GeneralOrbitNeuralLogEvidenceUpdater(_ReplayMixin, LogEvidenceUpdater):
    """Stateful finite-group neural updater with orbit-normalized log evidence."""

    def __init__(self, input_dim: int, model_factory=None, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh", min_train_samples: int = 10, update_epochs: int = 5, replay_batch_size: int = 32, lr: float = 1e-2, weight_decay: float = 1e-4, clip_value: float = 20.0, seed: int | None = 0):
        _require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.min_train_samples = int(min_train_samples)
        self.update_epochs = int(update_epochs)
        self.replay_batch_size = int(replay_batch_size)
        self.clip_value = float(clip_value)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.min_train_samples < 0 or self.update_epochs < 0 or self.replay_batch_size <= 0 or self.clip_value <= 0:
            raise ValueError("Invalid neural updater hyperparameter.")
        if seed is not None:
            torch.manual_seed(int(seed))
        self.model = _build_model(self.input_dim, model_factory, hidden_width, hidden_depth, activation).to("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.model.eval()

    def _check_vector(self, x) -> np.ndarray:
        vector = _as_vector(x)
        if vector.shape != (self.input_dim,):
            raise ValueError(f"x must have shape ({self.input_dim},).")
        return vector

    def _orbit_tensor(self, group, group_elements, vector: np.ndarray) -> torch.Tensor:
        orbit = [_as_vector(group.action(g, vector)) for g in group_elements]
        if any(v.shape != (self.input_dim,) for v in orbit):
            raise ValueError(f"Group action must return vectors with shape ({self.input_dim},).")
        return torch.as_tensor(np.stack(orbit, axis=0), dtype=torch.float32)

    def get_log_evidence(self, group, x, rng) -> float:
        vector = self._check_vector(x)
        group_elements = list(group.elements())
        if not group_elements:
            raise ValueError("group.elements() must return at least one element.")
        if len(self.observations) < self.min_train_samples:
            return 0.0
        with torch.no_grad():
            tensor = torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)
            observed_score = torch.clamp(_model_scores(self.model, tensor)[0], -self.clip_value, self.clip_value)
            orbit_scores = torch.clamp(_model_scores(self.model, self._orbit_tensor(group, group_elements, vector)), -self.clip_value, self.clip_value)
            log_mean_exp = torch.logsumexp(orbit_scores, dim=0) - math.log(len(group_elements))
            return float((observed_score - log_mean_exp).item())

    def update(self, G, x, rng):
        vector = self._check_vector(x)
        super().update(G, vector.copy(), rng)
        if len(self.observations) < self.min_train_samples or self.update_epochs == 0:
            return
        group_elements = list(G.elements())
        if not group_elements:
            raise ValueError("group.elements() must return at least one element.")
        self.model.train()
        for _ in range(self.update_epochs):
            batch = self._sample_replay_batch(rng)
            batch_tensor = torch.as_tensor(batch, dtype=torch.float32)
            orbit_batch = [[_as_vector(G.action(g, sample)) for g in group_elements] for sample in batch]
            orbit_tensor = torch.as_tensor(np.asarray(orbit_batch), dtype=torch.float32).reshape(-1, self.input_dim)
            self.optimizer.zero_grad()
            observed_scores = _model_scores(self.model, batch_tensor)
            orbit_scores = _model_scores(self.model, orbit_tensor).reshape(batch_tensor.shape[0], len(group_elements))
            log_mean_exp = torch.logsumexp(orbit_scores, dim=1) - math.log(len(group_elements))
            loss = -(observed_scores - log_mean_exp).mean()
            loss.backward()
            self.optimizer.step()
        self.model.eval()


class SignFlipNeuralLogEvidenceUpdater(_ReplayMixin, LogEvidenceUpdater):
    """Stateful sign-flip neural updater."""

    def __init__(self, input_dim: int, model_factory=None, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh", min_train_samples: int = 10, update_epochs: int = 5, replay_batch_size: int = 32, lr: float = 1e-2, weight_decay: float = 1e-4, clip_value: float = 20.0, seed: int | None = 0):
        _require_torch()
        super().__init__()
        self.input_dim = int(input_dim)
        self.min_train_samples = int(min_train_samples)
        self.update_epochs = int(update_epochs)
        self.replay_batch_size = int(replay_batch_size)
        self.clip_value = float(clip_value)
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if self.min_train_samples < 0 or self.update_epochs < 0 or self.replay_batch_size <= 0 or self.clip_value <= 0:
            raise ValueError("Invalid neural updater hyperparameter.")
        if seed is not None:
            torch.manual_seed(int(seed))
        self.model = _build_model(self.input_dim, model_factory, hidden_width, hidden_depth, activation).to("cpu")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.model.eval()

    def _check_vector(self, x) -> np.ndarray:
        vector = _as_vector(x)
        if vector.shape != (self.input_dim,):
            raise ValueError(f"x must have shape ({self.input_dim},).")
        return vector

    def _log_ratio_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return _model_scores(self.model, tensor) - _model_scores(self.model, -tensor)

    def get_log_evidence(self, group, x, rng) -> float:
        vector = self._check_vector(x)
        if len(self.observations) < self.min_train_samples:
            return 0.0
        with torch.no_grad():
            tensor = torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)
            log_ratio = torch.clamp(self._log_ratio_tensor(tensor)[0], -self.clip_value, self.clip_value)
            return float((math.log(2.0) - F.softplus(-log_ratio)).item())

    def update(self, G, x, rng):
        vector = self._check_vector(x)
        super().update(G, vector.copy(), rng)
        if len(self.observations) < self.min_train_samples or self.update_epochs == 0:
            return
        self.model.train()
        for _ in range(self.update_epochs):
            batch_tensor = torch.as_tensor(self._sample_replay_batch(rng), dtype=torch.float32)
            self.optimizer.zero_grad()
            loss = F.softplus(-self._log_ratio_tensor(batch_tensor)).mean()
            loss.backward()
            self.optimizer.step()
        self.model.eval()


class GeneralOrbitNeuralT(DynamicT):
    """Legacy stateless DynamicT wrapper for finite-group neural orbit contrast."""

    def __init__(self, past_data: list[np.ndarray], group, model_factory=None, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh", min_train_samples: int = 10, epochs: int = 100, lr: float = 1e-2, weight_decay: float = 1e-4, clip_value: float = 20.0, seed: int | None = 0):
        _require_torch()
        self.group = group
        self.clip_value = float(clip_value)
        self.neutral = len(past_data) < int(min_train_samples)
        self.input_dim = None
        self.model = None
        if self.neutral:
            return
        samples = _stack_past_data(past_data)
        if samples is None:
            self.neutral = True
            return
        self.input_dim = samples.shape[1]
        group_elements = list(group.elements())
        if seed is not None:
            torch.manual_seed(int(seed))
        self.model = _build_model(self.input_dim, model_factory, hidden_width, hidden_depth, activation).to("cpu")
        samples_tensor = torch.as_tensor(samples, dtype=torch.float32)
        orbit_samples = [[_as_vector(group.action(g, sample)) for g in group_elements] for sample in samples]
        orbit_tensor = torch.as_tensor(np.asarray(orbit_samples), dtype=torch.float32).reshape(-1, self.input_dim)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.model.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            observed_scores = _model_scores(self.model, samples_tensor)
            orbit_scores = _model_scores(self.model, orbit_tensor).reshape(samples_tensor.shape[0], len(group_elements))
            loss = -(observed_scores - (torch.logsumexp(orbit_scores, dim=1) - math.log(len(group_elements)))).mean()
            loss.backward()
            optimizer.step()
        self.model.eval()

    def score(self, x) -> float:
        if self.neutral:
            return 0.0
        vector = _as_vector(x)
        with torch.no_grad():
            raw_score = float(_model_scores(self.model, torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0))[0].item())
        return float(np.clip(raw_score, -self.clip_value, self.clip_value))

    def __call__(self, x) -> float:
        return float(math.exp(self.score(x)))


class SignFlipNeuralT(DynamicT):
    """Legacy stateless DynamicT wrapper for sign-flip neural orbit contrast."""

    def __init__(self, past_data: list[np.ndarray], model_factory=None, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh", min_train_samples: int = 10, epochs: int = 100, lr: float = 1e-2, weight_decay: float = 1e-4, clip_value: float = 20.0, seed: int | None = 0):
        _require_torch()
        self.clip_value = float(clip_value)
        self.neutral = len(past_data) < int(min_train_samples)
        self.input_dim = None
        self.model = None
        if self.neutral:
            return
        samples = _stack_past_data(past_data)
        if samples is None:
            self.neutral = True
            return
        self.input_dim = samples.shape[1]
        if seed is not None:
            torch.manual_seed(int(seed))
        self.model = _build_model(self.input_dim, model_factory, hidden_width, hidden_depth, activation).to("cpu")
        samples_tensor = torch.as_tensor(samples, dtype=torch.float32)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        self.model.train()
        for _ in range(int(epochs)):
            optimizer.zero_grad()
            log_ratio = _model_scores(self.model, samples_tensor) - _model_scores(self.model, -samples_tensor)
            loss = F.softplus(-log_ratio).mean()
            loss.backward()
            optimizer.step()
        self.model.eval()

    def log_ratio(self, x) -> float:
        if self.neutral:
            return 0.0
        vector = _as_vector(x)
        with torch.no_grad():
            tensor = torch.as_tensor(vector, dtype=torch.float32).unsqueeze(0)
            raw_ratio = float((_model_scores(self.model, tensor) - _model_scores(self.model, -tensor))[0].item())
        return float(np.clip(raw_ratio, -self.clip_value, self.clip_value))

    def __call__(self, x) -> float:
        return float(math.exp(0.5 * self.log_ratio(x)))


TORCH_AVAILABLE = torch is not None
TORCH_IMPORT_ERROR = _TORCH_IMPORT_ERROR

__all__ = ["GeneralOrbitNeuralLogEvidenceUpdater", "GeneralOrbitNeuralT", "SignFlipNeuralLogEvidenceUpdater", "SignFlipNeuralT", "TORCH_AVAILABLE", "TORCH_IMPORT_ERROR"]
