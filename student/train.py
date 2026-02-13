import argparse
import torch
import torch.nn as nn
import numpy as np
import wandb

from tqdm import tqdm

from student.Transformer import TransformerLM
from student.optimize import AdamW, cosine_annealing_scheduler, gradient_clipping, cross_entropy


def data_load(dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device):
    n = len(dataset)
    m = context_length

    random_starts = np.random.randint(0, n - m, size=batch_size)  # [B]
    offset = np.arange(m) # [m]
    indices = random_starts[:, None] + offset[None, :] # [B, m]

    inputs_np = dataset[indices]
    targets_np = dataset[indices + 1]

    inputs = torch.from_numpy(inputs_np).long().to(device)
    targets = torch.from_numpy(targets_np).long().to(device)
    return inputs, targets


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str):
    save = {
        "iteration": iteration,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict()
    }
    torch.save(save, out)


def load_checkpoint(src: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    state_dict = torch.load(src, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    iteration = state_dict["iteration"]
    return model, optimizer, iteration


def train_pipeline():
    # argparse
    parser = argparse.ArgumentParser()
    # dataset config
    parser.add_argument("--dataset_path", type=str, default="student/checkpoint/BPE/ids_train.npy")
    parser.add_argument("--vocab_size", type=int, default=10000)
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
    # training config
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    # scheduler
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--warm_up_steps", type=int, default=100)
    # optimizer
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)
    # save
    parser.add_argument("--out_path", type=str, default="student/checkpoint/model/model.pt")
    args = parser.parse_args()

    # wandb init
    wandb.init(
        project="LLM-A1-test",
        config=vars(args)
    )

    # determine device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # init model
    model = TransformerLM(vocab_size=args.vocab_size,
                          context_length=args.context_length,
                          num_layers=args.num_layers,
                          d_model=args.d_model,
                          num_heads=args.num_heads,
                          d_ff=args.d_ff,
                          rope=args.rope,
                          theta=args.theta,
                          eps=args.eps,
                          device=device,
                          dtype=args.dtype).to(device)
    model.train()

    # training config
    epochs = args.epochs
    iterations = args.iterations
    batch_size = args.batch_size
    context_length = args.context_length
    max_l2_norm = args.max_l2_norm

    # build optimizer
    optimizer = AdamW(params=model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)

    # scheduler
    lr_max = args.lr
    lr_min = args.lr_min
    warm_up_steps = args.warm_up_steps
    total_steps = epochs * iterations

    # load tokenzied data
    dataset = np.memmap(args.dataset_path, dtype=np.uint16, mode="r")

    # train loop
    global_step = 0
    for epoch in tqdm(range(epochs), leave=False, desc="Epoch:"):
        for itr in tqdm(range(iterations), leave=False, desc="Batch:"):
            inputs, targets = data_load(dataset, batch_size, context_length, device)

            # forward
            logits = model(inputs)

            # loss compute
            loss = cross_entropy(logits, targets)

            # gradient zeroing
            optimizer.zero_grad()

            # backward
            loss.backward()

            # gradient clip
            if max_l2_norm and max_l2_norm > 0:
                gradient_clipping(model.parameters(), max_l2_norm=max_l2_norm)

            # step
            optimizer.step()
            global_step += 1

            # lr scheduler
            lr = cosine_annealing_scheduler(
                global_step,
                lr_max,
                lr_min,
                warm_up_steps,
                total_steps
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # logging
            if global_step % 10 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "step": global_step,
                })

            # save model
            save_checkpoint(model, optimizer, global_step, args.out_path)
    return model


if __name__ == '__main__':
    train_pipeline()