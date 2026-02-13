import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.zeros(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3*std, b=3*std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.t()


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(num_embeddings, embedding_dim, device=device, dtype=dtype))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        a_square = x.pow(2)
        a_square_sum = a_square.sum(dim=-1, keepdim=True)
        rms = torch.sqrt((a_square_sum / self.d_model) + self.eps)
        return ((x / rms) * self.weight).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff

        std = math.sqrt(2.0 / (self.d_model + self.d_ff))
        self.weight1 = nn.Parameter(torch.zeros(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.weight2 = nn.Parameter(torch.zeros(self.d_model, self.d_ff, device=device, dtype=dtype))
        self.weight3 = nn.Parameter(torch.zeros(self.d_ff, self.d_model, device=device, dtype=dtype))

        torch.nn.init.trunc_normal_(self.weight1, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.weight2, mean=0.0, std=std, a=-3*std, b=3*std)
        torch.nn.init.trunc_normal_(self.weight3, mean=0.0, std=std, a=-3*std, b=3*std)

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        gate = self.SiLU(x @ self.weight1.t())
        up = gate * (x @ self.weight3.t())
        down = up @ self.weight2.t()
        return down


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        k = torch.arange(d_k // 2, device=device) # [d_k/2]
        self._theta = 1.0 / (self.theta ** ((2 * (k + 1) - 2) / self.d_k)) # [d_k/2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        x_even = x[..., 0::2] # [B, S, d/2]
        x_odd = x[..., 1::2] # [B, S, d/2]

        theta = token_positions.unsqueeze(-1) * self._theta # [B, S] -> [B, S, 1] -> [B, S, d_k/2]
        cos = torch.cos(theta)
        sin = torch.sin(theta)

        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        x_out = torch.zeros_like(x)
        x_out[..., 0::2] = x_even_rot
        x_out[..., 1::2] = x_odd_rot
        return x_out


def softmax(x: torch.Tensor, dim: int):
    max_val = torch.max(x, dim=dim, keepdim=True)[0] # [B, S, 1]
    exp = torch.exp(x - max_val) # [B, S, D]
    sum_exp = torch.sum(exp, dim=dim, keepdim=True) # [B, S, 1]
    return exp / sum_exp


def scaled_dot_product_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor, device: torch.device = None, dtype: torch.dtype = None):
    d_k = key.shape[-1]
    logits = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        logits = logits.masked_fill(~mask, float('-inf'))
    attention_score = softmax(logits, dim=-1)
    output = attention_score @ value
    return output

