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

        self.gate_proj = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.up_proj = Linear(in_features=self.d_model, out_features=self.d_ff, device=device, dtype=dtype)
        self.down_proj = Linear(in_features=self.d_ff, out_features=self.d_model, device=device, dtype=dtype)

    def SiLU(self, x: torch.Tensor):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        gate_out = self.SiLU(self.gate_proj(x))
        up_out = gate_out * self.up_proj(x)
        down_out = self.down_proj(up_out)
        return down_out


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


def softmax(x: torch.Tensor, dim: int, temp: float = 1.0):
    x_temp = x / temp
    max_val = torch.max(x_temp, dim=dim, keepdim=True)[0] # [B, S, 1]
    exp = torch.exp(x_temp - max_val) # [B, S, D]
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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: bool = False, theta: float = None, max_seq_len: int = None, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.num_heads = num_heads

        self.q_proj = Linear(in_features=self.d_model, out_features=self.d_k * self.num_heads, device=device, dtype=dtype)
        self.k_proj = Linear(in_features=self.d_model, out_features=self.d_k * self.num_heads, device=device, dtype=dtype)
        self.v_proj = Linear(in_features=self.d_model, out_features=self.d_v * self.num_heads, device=device, dtype=dtype)
        self.o_proj = Linear(in_features=self.d_v * self.num_heads, out_features=self.d_model, device=device, dtype=dtype)

        self.rope = rope
        if self.rope:
            assert theta is not None
            assert max_seq_len is not None
            self.rope_emb = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device)
        else:
            self.rope_emb = None

    def forward(self, x: torch.Tensor):
        input_shape = x.shape[:-1] # [B, S]
        hidden_shape = (*input_shape, self.num_heads, -1) # [B, S, H, d_h]

        query = self.q_proj(x).view(hidden_shape).transpose(1, 2) # [B, S, d_m] -> [B, S, H, d_h] -> [B, H, S, d_h]
        key = self.k_proj(x).view(hidden_shape).transpose(1, 2)
        value = self.v_proj(x).view(hidden_shape).transpose(1, 2)

        if self.rope:
            position = torch.arange(input_shape[-1], device=x.device).unsqueeze(0).expand(input_shape)
            query = self.rope_emb(query, position)
            key = self.rope_emb(key, position)

        mask = torch.triu(torch.ones(input_shape[-1], input_shape[-1], device=x.device, dtype=torch.bool), diagonal=1) # [S, S]
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, S, S]
        mask = ~mask

        attn_output = scaled_dot_product_attention(query, key, value, mask, x.device, x.dtype).transpose(1, 2) # [B, H, S, d_h] -> [B, S, H, d_h]
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, rope: bool = False, theta: float = None,
                 eps: float = 1e-5, max_seq_len: int = None, device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.self_attn = MultiHeadAttention(d_model=d_model,
                                            num_heads=num_heads,
                                            rope=rope,
                                            theta=theta,
                                            max_seq_len=max_seq_len,
                                            device=device,
                                            dtype=dtype)

        self.d_ff = d_ff
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        self.input_norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.post_atten_norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.input_norm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_atten_norm(x)
        x = self.ffn(x)
        x = residual + x
        return x


class TransformerLM(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, num_layers: int, d_model: int, num_heads: int,
                 d_ff: int, rope: bool = False, theta: float = None, eps: float = 1e-5,
                 device: torch.device = None, dtype: torch.dtype = None):
        super().__init__()
        self.emb = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model=d_model,
                              num_heads=num_heads,
                              d_ff=d_ff,
                              rope=rope,
                              theta=theta,
                              eps=eps,
                              max_seq_len=context_length,
                              device=device,
                              dtype=dtype)
            for _ in range(num_layers)]
        )
        self.norm = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor):
        embed = self.emb(token_ids)
        for layer in self.layers:
            embed = layer(embed)
        embed = self.norm(embed)
        logits = self.lm_head(embed)
        return logits
