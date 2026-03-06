import math
import torch
import triton
import triton.language as tl


def flash_attention_backward(Q, K, V, O, dO, L, is_causal=False):
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)

    D = (dO * O).sum(dim=-1)

    S = torch.einsum('bqd,bkd->bqk', Q, K) * scale

    if is_causal:
        idx = torch.arange(N, device=Q.device)
        causal_mask = idx[:, None] >= idx[None, :]
        S = torch.where(causal_mask[None], S, torch.tensor(-1e6, device=Q.device, dtype=S.dtype))

    P = torch.exp(S - L[:, :, None])

    dV = torch.einsum('bqk,bqd->bkd', P, dO)
    dP = torch.einsum('bqd,bkd->bqk', dO, V)

    dS = P * (dP - D[:, :, None])

    dQ = torch.einsum('bqk,bkd->bqd', dS, K) * scale
    dK = torch.einsum('bqk,bqd->bkd', dS, Q) * scale

    return dQ, dK, dV


class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        Br = 32
        Bc = 32

        O = torch.zeros_like(Q)
        L = torch.zeros(B, N, device=Q.device, dtype=Q.dtype)

        Tr = math.ceil(N / Br)
        Tc = math.ceil(N / Bc)

        for i in range(Tr):
            q_start = i * Br
            q_end = min(q_start + Br, N)
            Qi = Q[:, q_start:q_end, :]

            m_i = torch.full((B, q_end - q_start), float('-inf'), device=Q.device, dtype=Q.dtype)
            l_i = torch.zeros(B, q_end - q_start, device=Q.device, dtype=Q.dtype)
            O_i = torch.zeros(B, q_end - q_start, d, device=Q.device, dtype=Q.dtype)

            for j in range(Tc):
                k_start = j * Bc
                k_end = min(k_start + Bc, N)
                Kj = K[:, k_start:k_end, :]
                Vj = V[:, k_start:k_end, :]

                S_ij = torch.einsum('bqd,bkd->bqk', Qi, Kj) * scale

                if is_causal:
                    q_idx = torch.arange(q_start, q_end, device=Q.device)
                    k_idx = torch.arange(k_start, k_end, device=Q.device)
                    causal_mask = q_idx[:, None] >= k_idx[None, :]
                    S_ij = torch.where(causal_mask[None], S_ij, torch.tensor(float('-inf'), device=Q.device, dtype=Q.dtype))

                m_ij = S_ij.amax(dim=-1)
                m_new = torch.maximum(m_i, m_ij)

                P_ij = torch.exp(S_ij - m_new[:, :, None])
                correction = torch.exp(m_i - m_new)

                l_i = correction * l_i + P_ij.sum(dim=-1)
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
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None


@triton.jit
def flash_fwd_kernel(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qq, stride_qd,
    stride_kb, stride_kk, stride_kd,
    stride_vb, stride_vk, stride_vd,
    stride_ob, stride_oq, stride_od,
    stride_lb, stride_lq,
    N_QUERIES, N_KEYS,
    scale,
    D: tl.constexpr,
    Q_TILE_SIZE: tl.constexpr,
    K_TILE_SIZE: tl.constexpr,
    is_causal: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(
        Q_ptr + batch_index * stride_qb,
        shape=(N_QUERIES, D),
        strides=(stride_qq, stride_qd),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        K_ptr + batch_index * stride_kb,
        shape=(N_KEYS, D),
        strides=(stride_kk, stride_kd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        V_ptr + batch_index * stride_vb,
        shape=(N_KEYS, D),
        strides=(stride_vk, stride_vd),
        offsets=(0, 0),
        block_shape=(K_TILE_SIZE, D),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        O_ptr + batch_index * stride_ob,
        shape=(N_QUERIES, D),
        strides=(stride_oq, stride_od),
        offsets=(query_tile_index * Q_TILE_SIZE, 0),
        block_shape=(Q_TILE_SIZE, D),
        order=(1, 0),
    )
    L_block_ptr = tl.make_block_ptr(
        L_ptr + batch_index * stride_lb,
        shape=(N_QUERIES,),
        strides=(stride_lq,),
        offsets=(query_tile_index * Q_TILE_SIZE,),
        block_shape=(Q_TILE_SIZE,),
        order=(0,),
    )

    Q_i = tl.load(Q_block_ptr)
    O_i = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    m_i = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)

    # Query index vector for causal masking (absolute positions)
    q_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)  # (Q_TILE_SIZE,)

    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)
    for j in range(Tk):
        K_j = tl.load(K_block_ptr)
        V_j = tl.load(V_block_ptr)

        S_ij = tl.dot(Q_i, tl.trans(K_j)).to(tl.float32) * scale

        if is_causal:
            k_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)  # (K_TILE_SIZE,)
            causal_mask = q_offsets[:, None] >= k_offsets[None, :]    # (Q_TILE_SIZE, K_TILE_SIZE)
            S_ij = S_ij + tl.where(causal_mask, 0.0, -1e6)

        m_ij = tl.max(S_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)

        P_ij = tl.exp(S_ij - m_new[:, None])
        correction = tl.exp(m_i - m_new)

        l_i = correction * l_i + tl.sum(P_ij, axis=1)

        O_i = correction[:, None] * O_i
        O_i = tl.dot(P_ij.to(V_j.dtype), V_j, acc=O_i)

        m_i = m_new

        K_block_ptr = tl.advance(K_block_ptr, (K_TILE_SIZE, 0))
        V_block_ptr = tl.advance(V_block_ptr, (K_TILE_SIZE, 0))

    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    tl.store(O_block_ptr, O_i.to(O_block_ptr.type.element_ty))
    tl.store(L_block_ptr, L_i)


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, is_causal=False):
        B, N, d = Q.shape
        scale = 1.0 / math.sqrt(d)

        Q_TILE_SIZE = 64
        K_TILE_SIZE = 64

        O = torch.empty_like(Q)
        L = torch.empty(B, N, device=Q.device, dtype=torch.float32)

        Tq = triton.cdiv(N, Q_TILE_SIZE)
        grid = (Tq, B)

        flash_fwd_kernel[grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N, N,
            scale,
            D=d,
            Q_TILE_SIZE=Q_TILE_SIZE,
            K_TILE_SIZE=K_TILE_SIZE,
            is_causal=is_causal,
        )

        ctx.save_for_backward(Q, K, V, O, L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, L = ctx.saved_tensors
        dQ, dK, dV = flash_attention_backward(Q, K, V, O, dO, L, ctx.is_causal)
        return dQ, dK, dV, None