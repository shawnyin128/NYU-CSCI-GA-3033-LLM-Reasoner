import torch

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor):
    max_logits = torch.max(logits, dim=-1, keepdim=True)[0]
    logits_stable = logits - max_logits

    logsumexp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))

    target_logits = logits_stable.gather(
        dim=-1,
        index=targets.unsqueeze(-1)
    ).squeeze(-1)

    loss = -target_logits + logsumexp
    return loss.mean()

