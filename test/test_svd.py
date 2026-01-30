import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_printoptions(precision=5)
class LowRankLinear(nn.Module):
    """
    用 U @ V 替代原始 Linear
    """
    def __init__(self, W: torch.Tensor, b: torch.Tensor = None, rank: int = 512):
        super().__init__()
        # SVD 分解
        U, V = self.svd_low_rank(W, rank)
        self.U = nn.Parameter(U)
        self.V = nn.Parameter(V)
        if b is not None:
            self.bias = nn.Parameter(b)
        else:
            self.bias = None

    @staticmethod
    def svd_low_rank(W: torch.Tensor, rank: int):
        U_full, S_full, Vh_full = torch.linalg.svd(W, full_matrices=False)
        U = U_full[:, :rank]
        S = torch.diag(S_full[:rank])
        V = Vh_full[:rank, :]
        return U @ S, V  # W ≈ U @ V

    def forward(self, x):
        # x: (batch, seq_len, in_features)
        y = x @ self.V.T   # (batch, seq_len, rank)
        y = y @ self.U.T   # (batch, seq_len, out_features)
        if self.bias is not None:
            y = y + self.bias
        return y

# -------------------------------
# 示例：替换 Transformer FFN 层
# -------------------------------
class SimpleFFN(nn.Module):
    def __init__(self, d_model=1024, d_ff=3072):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))

# 模拟 FFN 层
d_model = 1024
d_ff = 3072
rank = 10

ffn = SimpleFFN(d_model, d_ff)
x = torch.randn(2, d_model)

# 原输出
y_orig = ffn(x)

# -------------------------------
# 替换最后线性层 W2
# -------------------------------
W2 = ffn.w2.weight.data  # (d_model, d_ff)
b2 = ffn.w2.bias.data

# 创建低秩替代层
ffn_lowrank = LowRankLinear(W2, b2, rank=rank)

# 替换 FFN 层最后线性
def ffn_forward_lowrank(x):
    return ffn_lowrank(ffn.activation(ffn.w1(x)))

y_lowrank = ffn_forward_lowrank(x)

# -------------------------------
# 验证压缩误差--相对误差
# -------------------------------
error = torch.norm(y_orig - y_lowrank) / torch.norm(y_orig)
print(f"Relative reconstruction error: {error:.4f}")

print("Original output shape:", y_orig.shape)
print("Low-rank output shape:", y_lowrank.shape)