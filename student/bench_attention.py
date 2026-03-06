import timeit
import torch
import a1_basics.model as a1model

BATCH_SIZE = 8
D_K_LIST = [16, 32, 64, 128]
SEQ_LEN_LIST = [256, 1024, 4096, 8192, 16384]
WARMUP = 10
STEPS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{'d_k':>6} {'seq_len':>8} {'fwd_ms':>10} {'mem_before_bwd_mb':>20} {'bwd_ms':>10}")
print("-" * 60)

for d_k in D_K_LIST:
    for seq_len in SEQ_LEN_LIST:
        try:
            Q = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
            K = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)
            V = torch.randn(BATCH_SIZE, seq_len, d_k, device=device, requires_grad=True)

            for _ in range(WARMUP):
                out = a1model.scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()

            fwd_times = []
            for _ in range(STEPS):
                t0 = timeit.default_timer()
                out = a1model.scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                fwd_times.append(timeit.default_timer() - t0)

            fwd_avg_ms = sum(fwd_times) / len(fwd_times) * 1000
            torch.cuda.synchronize()
            mem_before_bwd_mb = torch.cuda.memory_allocated() / 1024 ** 2

            grad_out = torch.ones_like(out)
            for _ in range(WARMUP):
                out = a1model.scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                out.backward(grad_out)
                torch.cuda.synchronize()

            bwd_times = []
            for _ in range(STEPS):
                out = a1model.scaled_dot_product_attention(Q, K, V)
                torch.cuda.synchronize()
                t0 = timeit.default_timer()
                out.backward(grad_out)
                torch.cuda.synchronize()
                bwd_times.append(timeit.default_timer() - t0)

            bwd_avg_ms = sum(bwd_times) / len(bwd_times) * 1000

            print(f"{d_k:>6} {seq_len:>8} {fwd_avg_ms:>10.3f} {mem_before_bwd_mb:>20.1f} {bwd_avg_ms:>10.3f}")

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{d_k:>6} {seq_len:>8} {'OOM':>10} {'OOM':>20} {'OOM':>10}")