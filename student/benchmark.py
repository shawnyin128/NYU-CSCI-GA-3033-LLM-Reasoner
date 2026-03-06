import argparse
import timeit

import torch

import a1_basics.model as a1model
import a1_basics.nn_utils as a1utils


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
parser.add_argument("--pass-type", choices=["forward", "forward_backward"], default="forward_backward")
parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
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

x = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)
y = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=device)

# warmup
for _ in range(args.warmup_steps):
    if args.pass_type == "forward":
        with torch.no_grad():
            model(x)
    else:
        model.zero_grad()
        logits = model(x)
        loss = a1utils.cross_entropy(logits, y)
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()

# timing
times = []
for _ in range(args.num_steps):
    t0 = timeit.default_timer()
    if args.pass_type == "forward":
        with torch.no_grad():
            model(x)
    else:
        model.zero_grad()
        logits = model(x)
        loss = a1utils.cross_entropy(logits, y)
        loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = timeit.default_timer()
    times.append(t1 - t0)

avg = sum(times) / len(times)
std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
print(f"[{args.model_size or 'custom'}] {args.pass_type} | avg: {avg * 1000:.2f} ms, std: {std * 1000:.2f} ms")