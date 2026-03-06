import math
import torch


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        Br = 32  # tile size for queries
        Bc = 32  # tile size for keys/values

        O = torch.zeros_like(Q)
        L = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        for i in range(Tr):
            q_start = i * Br
            q_end = min(q_start + Br, N)
            Qi = Q[:, q_start:q_end, :]  # (B, Br, d)

            m_i = torch.full((B, q_end - q_start), float('-inf'), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, q_end - q_start, device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros(B, q_end - q_start, d, device=Q.device, dtype=Q.dtype)

            for j in range(Tc):
                k_start = j * Bc
                k_end = min(k_start + Bc, N)
                Kj = K[:, k_start:k_end, :]  # (B, Bc, d)
                Vj = V[:, k_start:k_end, :]  # (B, Bc, d)

                S_ij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale  # (B, Br, Bc)

                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device)
                    k_idx = torch.arange(k_start, k_end, device=Q.device)
                    causal_mask = q_idx[:, None] >= k_idx[None, :]  # (Br, Bc)
                    S_ij = torch.where(causal_mask[None], S_ij, torch.tensor(float('-inf'), device=Q.device, dtype=Q.dtype))

                m_ij = S_ij.amax(dim=-1)               # (B, Br)
                m_new = torch.maximum(m_i, m_ij)       # (B, Br)

                P_ij = torch.exp(S_ij - m_new[:, :, None])          # (B, Br, Bc)
                correction = torch.exp(m_i - m_new)                  # (B, Br)

                l_i = correction * l_i + P_ij.sum(dim=-1)            # (B, Br)
                O_i = correction[:, :, None] * O_i + torch.einsum('bqk,bkd->bqd', P_ij, Vj)
                m_i = m_new

            O_i = O_i / l_i[:, :, None]
            L[:, q_start:q_end] = m_i + torch.log(l_i)
            O[:, q_start:q_end, :] = O_i

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        raise NotImplementedError