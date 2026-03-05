import argparse
import timeit

import torch

import a1_basics.model as a1model
import a1_basics.nn_utils as a1utils


parser = argparse.ArgumentParser()
parser.add_argument("--vocab-size", type=int, default=10000)
parser.add_argument("--context-length", type=int, default=128)
parser.add_argument("--d-model", type=int, default=768)
parser.add_argument("--num-layers", type=int, default=12)
parser.add_argument("--num-heads", type=int, default=12)
parser.add_argument("--d-ff", type=int, default=3072)
parser.add_argument("--rope-theta", type=float, default=10000.0)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument("--warmup-steps", type=int, default=10)
parser.add_argument("--num-steps", type=int, default=50)
parser.add_argument("--pass-type", choices=["forward", "forward_backward"], default="forward_backward")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args()

device = args.device

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
    if device == "cuda":
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
    if device == "cuda":
        torch.cuda.synchronize()
    t1 = timeit.default_timer()
    times.append(t1 - t0)

avg = sum(times) / len(times)
print(f"avg: {avg * 1000:.2f} ms, min: {min(times) * 1000:.2f} ms, max: {max(times) * 1000:.2f} ms")