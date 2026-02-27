import torch
import torch.nn.functional as F

def outcome_bce_logits(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, y)

def gps_gaussian_nll(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    var = sigma**2 + eps
    nll = 0.5 * torch.log(2.0 * torch.pi * var) + 0.5 * (t - mu) ** 2 / var
    return nll.mean()

def targeted_regularized_logits(logits, t, mu, sigma, epsilon):
    return logits + epsilon * ((t - mu) / (sigma + 1e-8))

def epsilon_penalty(epsilon: torch.Tensor, epsilon_l2: float = 1.0) -> torch.Tensor:
    return epsilon_l2 * (epsilon**2).mean()

def monotonicity_penalty(model, x_num, x_cat, t, delta: float, direction: str = "decreasing"):
    out0 = model(x_num, x_cat, t)
    out1 = model(x_num, x_cat, t + delta)

    p0 = torch.sigmoid(out0["logits"])
    p1 = torch.sigmoid(out1["logits"])

    if direction == "decreasing":
        viol = torch.relu(p1 - p0)
    elif direction == "increasing":
        viol = torch.relu(p0 - p1)
    else:
        raise ValueError("direction must be 'decreasing' or 'increasing'")

    return (viol**2).mean()
