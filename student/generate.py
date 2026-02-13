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
    pass