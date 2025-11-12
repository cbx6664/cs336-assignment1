import torch.nn as nn
import torch


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 1. create parameter tensor
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        # 2. init parameter
        torch.nn.init.trunc_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: (..., in_features)
        # shape of self.weight: (out_features, in_features)
        # shape of return: (..., out_features)
        return x @ self.weight.T
