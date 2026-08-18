from __future__ import annotations

_TORCH_IMPORT_ERROR: ImportError | None = None

try:
    import torch
except ImportError as exc:
    torch = None
    _TORCH_IMPORT_ERROR = exc

_PYTORCH_REQUIRED_MESSAGE = "Neural strategies require PyTorch. Install torch or disable neural strategies."


def _require_torch() -> None:
    if torch is None:
        raise ImportError(_PYTORCH_REQUIRED_MESSAGE) from _TORCH_IMPORT_ERROR


def _activation_module(name: str):
    activation = name.lower()
    if activation == "tanh":
        return torch.nn.Tanh()
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    raise ValueError("activation must be one of: 'tanh', 'relu', 'gelu'.")


if torch is None:
    class ScalarMLP:
        """Small scalar-output MLP h: R^d -> R."""
        def __init__(self, input_dim: int, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh"):
            _require_torch()
else:
    class ScalarMLP(torch.nn.Module):
        """Small scalar-output MLP h: R^d -> R."""
        def __init__(self, input_dim: int, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh"):
            super().__init__()
            input_dim = int(input_dim)
            hidden_width = int(hidden_width)
            hidden_depth = int(hidden_depth)
            if input_dim <= 0:
                raise ValueError("input_dim must be positive.")
            if hidden_width <= 0:
                raise ValueError("hidden_width must be positive.")
            if hidden_depth < 0:
                raise ValueError("hidden_depth must be non-negative.")
            layers: list[torch.nn.Module] = []
            in_features = input_dim
            for _ in range(hidden_depth):
                layers.append(torch.nn.Linear(in_features, hidden_width))
                layers.append(_activation_module(activation))
                in_features = hidden_width
            layers.append(torch.nn.Linear(in_features, 1))
            self.network = torch.nn.Sequential(*layers)

        def forward(self, x):
            if x.ndim == 1:
                x = x.unsqueeze(0)
            return self.network(x).squeeze(-1)


def build_scalar_mlp(input_dim: int, hidden_width: int = 32, hidden_depth: int = 1, activation: str = "tanh"):
    _require_torch()
    return ScalarMLP(input_dim=input_dim, hidden_width=hidden_width, hidden_depth=hidden_depth, activation=activation)


TORCH_AVAILABLE = torch is not None
TORCH_IMPORT_ERROR = _TORCH_IMPORT_ERROR

__all__ = ["ScalarMLP", "TORCH_AVAILABLE", "TORCH_IMPORT_ERROR", "build_scalar_mlp"]
