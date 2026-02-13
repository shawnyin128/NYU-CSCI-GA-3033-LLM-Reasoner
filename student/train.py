import torch
import torch.nn as nn
import numpy as np


def data_load(dataset: np.ndarray, batch_size: int, context_length: int, device: torch.device):
    dataset_ts = torch.from_numpy(dataset).to(device)
    n = dataset_ts.numel()
    m = context_length
    valid_bound = n - m
    offset = torch.arange(m, device=device) # [m]
    random_starts = torch.randint(0, valid_bound, (batch_size,), device=device) # [B]
    indices = random_starts.unsqueeze(-1) + offset.unsqueeze(0) # [B, m]
    shift_indices = indices.clone() + 1
    inputs = dataset_ts[indices]
    targets = dataset_ts[shift_indices]
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


if __name__ == '__main__':
    dataset = np.load("student/checkpoint/BPE/ids_valid.npy")
    data_load(dataset=dataset, batch_size=4, context_length=4, device=torch.device("cuda:0"))