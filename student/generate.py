import torch

def nucleus_decoding(q: torch.Tensor, p: float):
    sorted_q, sorted_indices = torch.sort(q, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_q, dim=-1)

    mask = cumulative > p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False

    sorted_q[mask] = 0.0
    sorted_q = sorted_q / sorted_q.sum(dim=-1, keepdim=True)
    sampled_sorted = torch.multinomial(sorted_q, num_samples=1)
    sampled = torch.gather(sorted_indices, -1, sampled_sorted)
    return sampled


def model_generation():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--theta", type=float, default=10000.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument(
        "--dtype",
        type=lambda x: getattr(torch, x),
        default=torch.float32,
        help="torch dtype, e.g. float32"
    )
    # generation config
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--eos_token_id", type=int, default=None)
    # checkpoint path
    parser.add_argument("--checkpoint_path", type=str, default="student/checkpoint/model/model.pt")
    # prompt
    parser.add_argument("--prompt", type=str, required=True)
    args = parser.parse_args()

    # decide device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # init model
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope=args.rope,
        theta=args.theta,
        eps=args.eps,
        device=device,
        dtype=torch.float32,
    ).to(device)

    # load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    # ===== tokenizer =====
    # 你已有 tokenizer 实现
    tokenizer = load_your_tokenizer_somehow()

    # ===== encode prompt =====
    input_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # ===== generation loop =====
    with torch.no_grad():
        for _ in range(args.max_new_tokens):

            # 裁剪 context window
            if input_ids.size(1) > args.context_length:
                input_ids = input_ids[:, -args.context_length:]

            logits = model(input_ids)
            next_token_logits = logits[:, -1, :]  # [B, V]

            # temperature
            probs = softmax(next_token_logits, dim=-1, temp=args.temperature)

            # nucleus sampling
            if args.top_p < 1.0:
                next_token = nucleus_decoding(probs, args.top_p)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if args.eos_token_id is not None:
                if (next_token == args.eos_token_id).all():
                    break

    # ===== decode =====
    generated_ids = input_ids.squeeze(0).tolist()
    text = tokenizer.decode(generated_ids)

    print("\n===== Generated Text =====\n")
    print(text)