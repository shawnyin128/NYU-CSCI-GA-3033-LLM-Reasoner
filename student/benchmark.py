import argparse
import timeit
import math
from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx

import a1_basics.model as a1model
import a1_basics.nn_utils as a1utils
import a1_basics.optimizer as a1optim

from torch import Tensor
from einops import einsum
from jaxtyping import Float, Bool, Int

from a1_basics.nn_utils import softmax


@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys    d_k"],
    V: Float[Tensor, " ... keys    d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    d_k = K.shape[-1]

    with nvtx.range("computing attention scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx.range("computing softmax"):
        attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension

    with nvtx.range("final matmul"):
        output = einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")
    return output


a1model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


MODEL_PRESETS = {
    "small": dict(d_model=768,  d_ff=3072,  num_layers=12, num_heads=12),
    "medium": dict(d_model=1024, d_ff=4096,  num_layers=24, num_heads=16),
    "large": dict(d_model=1280, d_ff=5120,  num_layers=36, num_heads=20),
    "xl": dict(d_model=1600, d_ff=6400,  num_layers=48, num_heads=25),
    "2.7B": dict(d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
}

parser = argparse.ArgumentParser()
parser.add_argument("--model-size", choices=MODEL_PRESETS.keys(), default=None)
parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=128)
parser.add_argument("--d-model", type=int, default=768)
parser.add_argument("--num-layers", type=int, default=12)
parser.add_argument("--num-heads", type=int, default=12)
parser.add_argument("--d-ff", type=int, default=3072)
parser.add_argument("--rope-theta", type=float, default=10000.0)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--warmup-steps", type=int, default=5)
parser.add_argument("--num-steps", type=int, default=10)
parser.add_argument("--pass-type", choices=["forward", "forward_backward", "train"], default="forward_backward")
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("--bf16", action="store_true")
parser.add_argument("--memory", action="store_true")
parser.add_argument("--compile", action="store_true")
args = parser.parse_args()

if args.model_size is not None:
    for k, v in MODEL_PRESETS[args.model_size].items():
        setattr(args, k.replace("-", "_"), v)

device = torch.device(args.device)

model = a1model.BasicsTransformerLM(
    vocab_size=args.vocab_size,
    context_length=args.context_length,
    d_model=args.d_model,
    num_layers=args.num_layers,
    num_heads=args.num_heads,
    d_ff=args.d_ff,
    rope_theta=args.rope_theta,
).to(device)

if args.compile:
    model = torch.compile(model)

x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

optimizer = None
if args.pass_type == "train":
    optimizer = a1optim.AdamW(model.parameters(), lr=1e-4)

autocast_ctx = torch.autocast(device.type, dtype=torch.bfloat16) if args.bf16 else nullcontext()

# warmup
for _ in range(args.warmup_steps):
    with nvtx.range("warmup"):
        if args.pass_type == "forward":
            with torch.no_grad(), autocast_ctx:
                model(x)
        elif args.pass_type == "forward_backward":
            model.zero_grad()
            with autocast_ctx:
                logits = model(x)
                loss = a1utils.cross_entropy(logits, y)
            loss.backward()
        else:
            optimizer.zero_grad()
            with autocast_ctx:
                logits = model(x)
                loss = a1utils.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()

# timing
tag = f"{args.model_size or 'custom'}_ctx{args.context_length}"
fwd_times = []
bwd_times = []
opt_times = []
for _ in range(args.num_steps):
    with nvtx.range("step"):
        if args.pass_type == "forward":
            if args.memory:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            t0 = timeit.default_timer()
            with torch.no_grad(), autocast_ctx:
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd_times.append(timeit.default_timer() - t0)
            if args.memory:
                torch.cuda.memory._dump_snapshot(f"./memory_{tag}_forward.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
        elif args.pass_type == "forward_backward":
            if args.memory:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            model.zero_grad()
            t0 = timeit.default_timer()
            with nvtx.range("forward"), autocast_ctx:
                logits = model(x)
                loss = a1utils.cross_entropy(logits, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = timeit.default_timer()
            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
            if args.memory:
                torch.cuda.memory._dump_snapshot(f"./memory_{tag}_forward_backward.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
        else:
            if args.memory:
                torch.cuda.memory._record_memory_history(max_entries=1000000)
            optimizer.zero_grad()
            t0 = timeit.default_timer()
            with nvtx.range("forward"), autocast_ctx:
                logits = model(x)
                loss = a1utils.cross_entropy(logits, y)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = timeit.default_timer()
            with nvtx.range("backward"):
                loss.backward()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t2 = timeit.default_timer()
            with nvtx.range("optimizer"):
                optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            t3 = timeit.default_timer()
            fwd_times.append(t1 - t0)
            bwd_times.append(t2 - t1)
            opt_times.append(t3 - t2)
            if args.memory:
                torch.cuda.memory._dump_snapshot(f"./memory_{tag}_train.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)


def stats(times):
    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg * 1000, std * 1000

tag = f"{args.model_size or 'custom'}{'|bf16' if args.bf16 else ''}{'|compiled' if args.compile else ''}"
if args.pass_type == "forward":
    avg, std = stats(fwd_times)
    print(f"[{tag}] forward   | avg: {avg:.2f} ms, std: {std:.2f} ms")
elif args.pass_type == "forward_backward":
    fa, fs = stats(fwd_times)
    ba, bs = stats(bwd_times)
    print(f"[{tag}] forward   | avg: {fa:.2f} ms, std: {fs:.2f} ms")
    print(f"[{tag}] backward  | avg: {ba:.2f} ms, std: {bs:.2f} ms")
else:
    fa, fs = stats(fwd_times)
    ba, bs = stats(bwd_times)
    oa, os_ = stats(opt_times)
    print(f"[{tag}] forward   | avg: {fa:.2f} ms, std: {fs:.2f} ms")
    print(f"[{tag}] backward  | avg: {ba:.2f} ms, std: {bs:.2f} ms")
    print(f"[{tag}] optimizer | avg: {oa:.2f} ms, std: {os_:.2f} ms")