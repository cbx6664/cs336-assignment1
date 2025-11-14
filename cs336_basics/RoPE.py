import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # θ_{k} = Θ^{-(2k-2)/d} for k ∈ {1,...,d/2}
        i = torch.arange(0, d_k, 2, dtype=torch.float32)
        # 这里i已经等价于公式中的2k-2
        freqs = theta ** (-i / d_k)

        # pre compute angles for all possible positions[0, max_seq_len)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, freqs)  # (max_seq_len, d_k/2)

        # pre compute cos and sin
        cos_cached = torch.cos(angles)
        sin_cached = torch.sin(angles)
        self.register_buffer("cos_cached", cos_cached, persistent=False)
        self.register_buffer("sin_cached", sin_cached, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # 1. get cos/sin according to positions
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. reshape x
        # x.shape:(2,3,8) [batch_size, seq_len, d_k]
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        # 等价于: x.reshape(2, 3, -1, 2), *x.shape[:-1]是解包, =2,3; -1代表自动计算这个dim的值, 最后的2代表每对两个元素
        # x_reshaped:(batch_size, seq_len, 对数, 每对2个元素 )

        # 3.分离偶数位和奇数位
        # x' = x × cos(θ) - y × sin(θ)
        # y' = x × sin(θ) + y × cos(θ)
        x_even = x_reshaped[..., 0]
        x_odd = x_reshaped[..., 1]

        # 4. rotate
        x_even_rotated = x_even * cos - x_odd * sin
        x_odd_rotated = x_even * sin + x_odd * cos

        # 5. restore shape
        x_rotated = torch.stack([x_even_rotated, x_odd_rotated], dim=-1)
        # x_reshaped:(batch_size, seq_len, 对数, 每对2个元素) -> (batch_size, seq_len, d_k)
        x_out = x_rotated.reshape(*x.shape)
        return x_out
