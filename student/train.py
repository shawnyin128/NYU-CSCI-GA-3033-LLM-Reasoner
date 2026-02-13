import math
import torch
import torch.optim as optim

from typing import Optional, Callable, Iterable

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


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 0)  # Get iteration number from the state, or initial value.
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad  # Update weight tensor in-place.
                state["t"] = t + 1  # Increment iteration number.
        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 1
                    state["exp_avg"] = torch.zeros_like(param.data)
                    state["exp_avg_sq"] = torch.zeros_like(param.data)
                grad = param.grad.data

                step = state["step"]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # update m
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # update v
                exp_avg_sq.mul_(beta2).add_(grad**2, alpha=1 - beta2)

                # adjust lr
                lr_t = lr * math.sqrt(1 - beta2 ** step) / (1 - beta1 ** step)

                # update weight
                param.data -= lr_t * (exp_avg / (torch.sqrt(exp_avg_sq) + eps))

                # apply weight decay
                if weight_decay != 0:
                    param.data -= lr * weight_decay * param.data

                state["step"] += 1
        return loss


def cosine_annealing_scheduler(iter: int, lr_max: float, lr_min: float, warm_up_steps: int, cosine_annealing_steps: int):
    if iter < warm_up_steps:
        return (iter / warm_up_steps) * lr_max
    elif iter > cosine_annealing_steps:
        return lr_min
    else:
        return lr_min + 0.5 * (1 + math.cos(math.pi * (iter - warm_up_steps) / (cosine_annealing_steps - warm_up_steps))) * (lr_max - lr_min)


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), device="cuda"))
    opt = SGD([weights], lr=1e3)
    for t in range(10):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights ** 2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
