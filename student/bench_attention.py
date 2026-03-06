import timeit
import torch
import a1_basics.model as a1model

BATCH_SIZE = 8
D_K_LIST = [16, 32, 64, 128]
SEQ_LEN_LIST = [256, 1024, 4096, 8192, 16384]
WARMUP = 10
STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

compiled_attention = torch.compile(a1model.scaled_dot_product_attention)


def bench(attn_fn, d_k, seq_len):
    Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
    K = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
    V = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)

    # warmup forward
    for _ in range(WARMUP):
        out = attn_fn(Q, K, V)
        torch.cuda.synchronize()

    # time forward
    fwd_times = []
    for _ in range(STEPS):
        t0 = timeit.default_timer()
        out = attn_fn(Q, K, V)
        torch.cuda.synchronize()
        fwd_times.append(timeit.default_timer() - t0)

    fwd_avg_ms = sum(fwd_times) / len(fwd_times) * 1000
    mem_before_bwd_mb = torch.cuda.memory_allocated() / 1024 ** 2

    grad_out = torch.ones_like(out)

    for _ in range(WARMUP):
        out = attn_fn(Q, K, V)
        torch.cuda.synchronize()
        out.backward(grad_out)
        torch.cuda.synchronize()

    bwd_times = []
    for _ in range(STEPS):
        out = attn_fn(Q, K, V)
        torch.cuda.synchronize()
        t0 = timeit.default_timer()
        out.backward(grad_out)
        torch.cuda.synchronize()
        bwd_times.append(timeit.default_timer() - t0)

    bwd_avg_ms = sum(bwd_times) / len(bwd_times) * 1000
    return fwd_avg_ms, mem_before_bwd_mb, bwd_avg_ms


header = f"{'d_k':>6} {'seq_len':>8} | {'fwd(ms)':>10} {'mem(MB)':>10} {'bwd(ms)':>10} | {'fwd_c(ms)':>10} {'mem_c(MB)':>10} {'bwd_c(ms)':>10}"
print(header)
print("-" * len(header))

for d_k in D_K_LIST:
    for seq_len in SEQ_LEN_LIST:
        # uncompiled
        try:
            fwd, mem, bwd = bench(a1model.scaled_dot_product_attention, d_k, seq_len)
            base = f"{fwd:>10.3f} {mem:>10.1f} {bwd:>10.3f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            base = f"{'OOM':>10} {'OOM':>10} {'OOM':>10}"

        # compiled
        try:
            fwd_c, mem_c, bwd_c = bench(compiled_attention, d_k, seq_len)
            comp = f"{fwd_c:>10.3f} {mem_c:>10.1f} {bwd_c:>10.3f}"
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            comp = f"{'OOM':>10} {'OOM':>10} {'OOM':>10}"

        print(f"{d_k:>6} {seq_len:>8} | {base} | {comp}")