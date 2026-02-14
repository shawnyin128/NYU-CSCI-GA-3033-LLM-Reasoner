import torch
import argparse

from student.byte_pair_encoding import Tokenizer
from student.model import TransformerLM


def model_generation():
    parser = argparse.ArgumentParser()
    # model config
    parser.add_argument("--vocab_size", type=int, default=10000)
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
    parser.add_argument("--eos_token_id", type=int, default=0)
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

    # tokenizer
    tokenizer = Tokenizer.from_files("student/checkpoint/BPE", "student/checkpoint/BPE", special_tokens=["<|endoftext|>"])

    # tokenize prompt
    input_ids = tokenizer.encode(args.prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    # generate loop
    with torch.no_grad():
        generated = model.generate(input_ids, max_new_tokens=args.max_new_tokens, temp=args.temperature, top_p=args.top_p, eos_token_id=args.eos_token_id)

    generated_ids = generated.squeeze(0).tolist()
    text = tokenizer.decode(generated_ids)

    print(text)


if __name__ == "__main__":
    model_generation()