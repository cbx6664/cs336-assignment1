import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        # shape of weight: (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # shape of x: (batch_size, sequence_length, d_model)
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_square = x * x
        mean = torch.mean(x_square, dim=-1, keepdim=True)
        rms = torch.sqrt(mean + self.eps)
        rmsnorm = x / rms * self.weight
        return rmsnorm.to(in_dtype)  # 需要 return！
