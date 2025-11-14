import torch.nn as nn
from cs336_basics.Linear import Linear
import torch
import torch.nn.functional as F


class SwiGlu(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w1_out = self.w1(x)
        silu_out = F.silu(w1_out)
        w3_out = self.w3(x)
        gate_out = silu_out * w3_out
        output = self.w2(gate_out)

        return output
