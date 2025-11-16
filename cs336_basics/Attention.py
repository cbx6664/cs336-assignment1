import torch
import math
from cs336_basics.Softmax import softmax


def scaled_dot_product_attention(
    key: torch.Tensor,
    query: torch.Tensor,
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
