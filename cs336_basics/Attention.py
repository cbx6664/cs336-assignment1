import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int
import math
from cs336_basics.Softmax import softmax
from einops import rearrange
from cs336_basics.RoPE import RoPE


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Attention(Q, K, V) = softmax(Q.T @ K / sqrt(d_k)) @ V
    """
    # shape of queries and keys: (batch_size, ..., seq_len, d_k)
    # Q.T @ K 只是推导公式时的col-based写法, code时是row-based, 应该是Q @ K.T, (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)
    d_k = query.size(dim=-1)
    # tensor.T 在multi-dim时, 会反转所有dim, (0, 1, 2) → (2, 1, 0), 所以应该用.transpose(-2, -1)反转最后两个dim
    pre_softmax = query @ key.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(mask == False, float("-inf"))
    # shape of value: (batch_size, ..., seq_len, d_v)
    return softmax(pre_softmax, dim=-1) @ value


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        q_proj_weight: Float[Tensor, " hd_k d_in"],
        k_proj_weight: Float[Tensor, " hd_k d_in"],
        v_proj_weight: Float[Tensor, " hd_v d_in"],
        o_proj_weight: Float[Tensor, " d_model hd_v"],
        max_seq_len: int | None = None,
        theta: float | None = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj_weight = nn.Parameter(q_proj_weight)
        self.k_proj_weight = nn.Parameter(k_proj_weight)
        self.v_proj_weight = nn.Parameter(v_proj_weight)
        self.o_proj_weight = nn.Parameter(o_proj_weight)
        self.max_seq_len = max_seq_len
        self.theta = theta

    def forward(
        self,
        x: Float[Tensor, "batch_size seq_len d_model"],
        token_positions: Int[Tensor, " ... sequence_length"] | None = None,
    ) -> Float[Tensor, "batch_size seq_len d_model"]:
        # (batch_size, seq_len, d_model) @ (d_in, hd_k) -> (batch_size, seq_len, h*d_k)
        Q = x @ self.q_proj_weight.T
        K = x @ self.k_proj_weight.T
        V = x @ self.v_proj_weight.T

        # reshape tensors to seperate heads
        # (batch_size, seq_len, h*d_k) -> (batch_size, num_heads, seq_len, d_k)
        Q = rearrange(
            Q,
            "batch_size seq_len (h d_k) -> batch_size h seq_len d_k",
            h=self.num_heads,
        )

        K = rearrange(
            K,
            "batch_size seq_len (h d_k) -> batch_size h seq_len d_k",
            h=self.num_heads,
        )

        V = rearrange(
            V,
            "batch_size seq_len (h d_v) -> batch_size h seq_len d_v",
            h=self.num_heads,
        )

        # construct causal masking
        batch_size, seq_len, d_model = x.shape
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # apply RoPE (if theta and positions are provided)
        if self.theta is not None and token_positions is not None:
            d_k = d_model / self.num_heads
            rope = RoPE(self.theta, d_k, self.max_seq_len)
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)

        attn_output = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        # concat heads
        attn_output = rearrange(attn_output, "b h s d_v -> b s (h d_v)")

        # O proj
        output = attn_output @ self.o_proj_weight.T
        return output
