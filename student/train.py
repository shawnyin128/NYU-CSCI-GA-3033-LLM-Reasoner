import torch
import torch.nn as nn
import numpy as np

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

    inputs = torch.from_numpy(inputs_np).to(device)
    targets = torch.from_numpy(targets_np).to(device)
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
    args = parser.parse_args()

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

    # build optimizer
    optimizer = AdamW(params=model.parameters(),
                      lr=args.lr,
                      weight_decay=args.weight_decay)

    # load tokenzied data
    dataset = np.memmap(args.dataset_path, dtype=np.uint16, mode="r")

    # training
    epochs = args.epochs
    iterations = args.iterations
    batch_size = args.batch_size
    context_length = args.context_length
    max_l2_norm = args.max_l2_norm
    for i in range(epochs):
        for j in range(iterations):
            inputs, targets = data_load(dataset, batch_size, context_length, device)

            # forward
            logits = model(inputs)

            # loss compute
            loss = cross_entropy(logits, targets)

            # gradient zeroing
            optimizer.zero_grad()

            # backward
            loss.backward()

            # step
            optimizer.step()
    return model


if __name__ == '__main__':
    dataset = np.load("student/checkpoint/BPE/ids_valid.npy")
    data_load(dataset=dataset, batch_size=4, context_length=4, device=torch.device("cuda:0"))