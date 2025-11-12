import torch.nn as nn
import torch


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # 1. create parameter tensor
        self.emb_matrix = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        )

        # 2. init parameter
        torch.nn.init.trunc_normal_(self.emb_matrix)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # shape of token_ids: (...)  任意形状，例如 (batch, seq_len)
        # shape of self.emb_matrix: (num_embeddings, embedding_dim)
        # shape of output: (..., embedding_dim)

        # 直接用 token_ids 作为索引，从 embedding 矩阵中取出对应的向量
        return self.emb_matrix[token_ids]


"""
形状变化总结
    输入 token_ids 形状	Embedding 矩阵形状	输出形状
    () 单个 token	(V, D)	(D,)
    (S,) 序列	(V, D)	(S, D)
    (B, S) 批量序列	(V, D)	(B, S, D)
    (B, S, T) 3D	(V, D)	(B, S, T, D)
    其中 V=num_embeddings, D=embedding_dim
    
# 示例
matrix = torch.tensor([
    [1, 2, 3],  # row 0
    [4, 5, 6],  # row 1
    [7, 8, 9],  # row 2
])

# 单个索引
indices = torch.tensor(1)
result = matrix[indices]  # [4, 5, 6]

# 多个索引
indices = torch.tensor([0, 2, 1])
result = matrix[indices]
# 结果:
# [[1, 2, 3],
#  [7, 8, 9],
#  [4, 5, 6]]

# 批量索引（2D）
indices = torch.tensor([[0, 1], [2, 0]])
result = matrix[indices]
# 形状: (2, 2, 3)
"""
