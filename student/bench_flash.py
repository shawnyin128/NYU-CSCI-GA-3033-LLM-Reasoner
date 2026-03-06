import math
import torch
import triton
import triton.testing

from student.flash_attention import flash_fwd_kernel, flash_attention_backward

BATCH_SIZE = 1
SEQ_LENS   = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
D_SIZES    = [16, 32, 64, 128]
DTYPES     = [torch.bfloat16, torch.float32]


def get_tile_size(seq_len):
    for t in [128, 64, 32, 16]:
        if seq_len >= t:
            return t
    return 16


def triton_fwd(Q, K, V):
    B, N, d = Q.shape
    T = get_tile_size(N)
    O = torch.empty_like(Q)
    L = torch.empty(B, N, device=Q.device, dtype=torch.float32)
    flash_fwd_kernel[(triton.cdiv(N, T), B)](
        Q, K, V, O, L,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        O.stride(0), O.stride(1), O.stride(2),
        L.stride(0), L.stride(1),
        N, N, 1.0 / math.sqrt(d),
        D=d, Q_TILE_SIZE=T, K_TILE_SIZE=T, is_causal=True,
    )
    return O, L

def pytorch_fwd(Q, K, V):
    B, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    S = torch.einsum('bqd,bkd->bqk', Q, K) * scale
    idx = torch.arange(N, device=Q.device)
    mask = idx[:, None] >= idx[None, :]
    S = torch.where(mask[None], S, torch.tensor(float('-inf'), device=Q.device, dtype=S.dtype))
    P = torch.softmax(S, dim=-1)
    return torch.einsum('bqk,bkd->bqd', P, V)

def bench_one(seq_len, d, dtype):
    device = 'cuda'
    Q = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)
    K = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)
    V = torch.randn(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)

    results = {}

    try:
        O_t, L_t = triton_fwd(Q, K, V)
        dO = torch.ones_like(O_t)

        fwd_ms  = triton.testing.do_bench(lambda: triton_fwd(Q, K, V))
        bwd_ms  = triton.testing.do_bench(lambda: flash_attention_backward(Q, K, V, O_t, dO, L_t, True))
        e2e_ms  = triton.testing.do_bench(lambda: flash_attention_backward(
                      Q, K, V, *triton_fwd(Q, K, V)[:], dO if False else torch.ones_like(triton_fwd(Q, K, V)[0]), True))

        def triton_e2e():
            O_, L_ = triton_fwd(Q, K, V)
            flash_attention_backward(Q, K, V, O_, torch.ones_like(O_), L_, True)
        e2e_ms = triton.testing.do_bench(triton_e2e)

        results['triton'] = (fwd_ms, bwd_ms, e2e_ms)
    except Exception as e:
        results['triton'] = ('OOM', 'OOM', 'OOM')

    try:
        Q_pt = Q.clone().requires_grad_(True)
        K_pt = K.clone().requires_grad_(True)
        V_pt = V.clone().requires_grad_(True)

        O_pt = pytorch_fwd(Q_pt, K_pt, V_pt)
        dO_pt = torch.ones_like(O_pt)

        dO_pt = torch.ones(BATCH_SIZE, seq_len, d, device=device, dtype=dtype)

        fwd_ms_pt = triton.testing.do_bench(lambda: pytorch_fwd(Q, K, V))

        def pt_fwd_bwd():
            Q_b = Q.detach().requires_grad_(True)
            K_b = K.detach().requires_grad_(True)
            V_b = V.detach().requires_grad_(True)
            pytorch_fwd(Q_b, K_b, V_b).backward(dO_pt)

        e2e_ms_pt = triton.testing.do_bench(pt_fwd_bwd)

        def pt_bwd_only():
            Q_b = Q.detach().requires_grad_(True)
            K_b = K.detach().requires_grad_(True)
            V_b = V.detach().requires_grad_(True)
            out = pytorch_fwd(Q_b, K_b, V_b)
            torch.cuda.synchronize()
            return out, Q_b, K_b, V_b

        out_b, Q_b, K_b, V_b = pt_bwd_only()
        bwd_ms_pt = triton.testing.do_bench(lambda: out_b.backward(dO_pt, retain_graph=True))

        results['pytorch'] = (fwd_ms_pt, bwd_ms_pt, e2e_ms_pt)
    except Exception as e:
        results['pytorch'] = ('OOM', 'OOM', 'OOM')

    return results

def fmt(v):
    return f"{v:.3f}" if isinstance(v, float) else v

header = f"{'seq':>7} {'d':>4} {'dtype':>10} | {'triton_fwd':>12} {'triton_bwd':>12} {'triton_e2e':>12} | {'pt_fwd':>10} {'pt_bwd':>10} {'pt_e2e':>10}"
print(header)
print('-' * len(header))

for dtype in DTYPES:
    for d in D_SIZES:
        for seq_len in SEQ_LENS:
            r = bench_one(seq_len, d, dtype)
            tf, tb, te = r['triton']
            pf, pb, pe = r['pytorch']
            dtype_str = 'bf16' if dtype == torch.bfloat16 else 'fp32'
            print(f"{seq_len:>7} {d:>4} {dtype_str:>10} | "
                  f"{fmt(tf):>12} {fmt(tb):>12} {fmt(te):>12} | "
                  f"{fmt(pf):>10} {fmt(pb):>10} {fmt(pe):>10}")