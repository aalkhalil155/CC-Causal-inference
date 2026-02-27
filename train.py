import importlib
import inspect
import os
import pathlib
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Support both execution modes:
# 1) python train.py
# 2) python -m <package>.train
if __package__ in (None, ""):
    repo_root = pathlib.Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    data_module = importlib.import_module("data")
    model_module = importlib.import_module("model")
    utils_module = importlib.import_module("utils")
else:
    data_module = importlib.import_module(f"{__package__}.data")
    model_module = importlib.import_module(f"{__package__}.model")
    utils_module = importlib.import_module(f"{__package__}.utils")

CausalDataset = data_module.CausalDataset
generate_synthetic = data_module.generate_synthetic
set_seed = utils_module.set_seed
get_device = utils_module.get_device

DragonNetContinuous = getattr(model_module, "DragonNetContinuous", None)
DragonNetContinuousAdvanced = getattr(model_module, "DragonNetContinuousAdvanced", None)

ADVANCED_DEFAULTS = {
    "use_t_mlp": False,
    "t_mlp_hidden": 64,
    "t_mlp_layers": 1,
    "use_gamma": False,
    "use_cat_offset": False,
    "cat_offset_hidden": 32,
}


# -----------------------------
# Config helpers
# -----------------------------
def find_config_path() -> pathlib.Path:
    """Find config.yaml from this file's directory upward."""
    start = pathlib.Path(__file__).resolve().parent
    for path in [start, *start.parents]:
        candidate = path / "config.yaml"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not find config.yaml near train.py")


def load_config():
    """Load YAML config from config.yaml."""
    config_path = find_config_path()
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def pick_model_class():
    """Pick an available model class in priority order."""
    if DragonNetContinuousAdvanced is not None:
        return DragonNetContinuousAdvanced
    if DragonNetContinuous is not None:
        return DragonNetContinuous
    raise ImportError(
        "No supported model class found in model.py. "
        "Expected DragonNetContinuousAdvanced or DragonNetContinuous."
    )


def build_model(dcfg, mcfg, device):
    """Instantiate available model class while filling compatible defaults."""
    model_cls = pick_model_class()

    kwargs = {
        "n_num": dcfg["n_num"],
        "cat_cardinalities": dcfg["cat_cardinalities"],
        "emb_dim": mcfg["emb_dim"],
        "d_hidden": mcfg["d_hidden"],
        "n_shared_layers": mcfg["n_shared_layers"],
        "dropout": mcfg["dropout"],
        "min_sigma": mcfg.get("min_sigma", 0.05),
    }

    sig = inspect.signature(model_cls.__init__)
    for name, param in sig.parameters.items():
        if name == "self" or name in kwargs:
            continue

        if name in mcfg:
            kwargs[name] = mcfg[name]
            continue

        if name in ADVANCED_DEFAULTS:
            kwargs[name] = ADVANCED_DEFAULTS[name]
            continue

        if param.default is inspect.Parameter.empty:
            raise TypeError(
                f"Missing required model config parameter: '{name}'. "
                "Add it under config['model'] in config.yaml."
            )

    return model_cls(**kwargs).to(device)


def model_forward(model, batch):
    """Call model.forward with args required by the active model variant."""
    sig = inspect.signature(model.forward)
    forward_kwargs = {}
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in batch:
            forward_kwargs[name] = batch[name]
            continue
        if param.default is inspect.Parameter.empty:
            raise TypeError(f"Model forward requires '{name}' but it is missing from the batch.")
    return model(**forward_kwargs)


# -----------------------------
# Loss components
# -----------------------------
def outcome_bce_logits(y: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.binary_cross_entropy_with_logits(logits, y)


def gps_gaussian_nll(t: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """
    Negative log-likelihood for Gaussian: t ~ N(mu, sigma^2)
    """
    eps = 1e-8
    var = sigma**2 + eps
    nll = 0.5 * torch.log(2.0 * torch.pi * var) + 0.5 * (t - mu) ** 2 / var
    return nll.mean()


def targeted_regularized_logits(
    logits: torch.Tensor,
    t: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """
    DragonNet-style targeted regularization:
      logits_tilde = logits + epsilon * ((t - mu)/sigma)
    """
    return logits + epsilon * ((t - mu) / (sigma + 1e-8))


def epsilon_penalty(epsilon: torch.Tensor, epsilon_l2: float = 1.0) -> torch.Tensor:
    return float(epsilon_l2) * (epsilon ** 2).mean()


@torch.no_grad()
def monotonicity_penalty(
    model,
    batch,
    out0,
    delta: float,
    direction: str = "decreasing",
) -> torch.Tensor:
    """
    Finite difference monotonicity penalty on p(y=1|x,t):
    - decreasing: p(t+delta) <= p(t)
    - increasing: p(t+delta) >= p(t)

    Only works if model.forward depends on 't' (i.e., batch has 't' and forward uses it).
    If model doesn't take t, returns 0.
    """
    # If model forward does not accept t, can't compute monotonicity wrt t
    sig = inspect.signature(model.forward)
    if "t" not in sig.parameters:
        return torch.tensor(0.0, device=out0["logits"].device)

    # Create batch with shifted treatment
    batch1 = dict(batch)
    batch1["t"] = batch["t"] + float(delta)

    out1 = model_forward(model, batch1)

    p0 = torch.sigmoid(out0["logits"])
    p1 = torch.sigmoid(out1["logits"])

    if direction == "decreasing":
        viol = torch.relu(p1 - p0)  # should be <= 0
    elif direction == "increasing":
        viol = torch.relu(p0 - p1)
    else:
        raise ValueError("mono_direction must be 'decreasing' or 'increasing'")

    return (viol ** 2).mean()


# -----------------------------
# Eval + checkpoint
# -----------------------------
def save_checkpoint_local(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


@torch.no_grad()
def evaluate(model, loader, device, epsilon, lcfg):
    model.eval()
    losses = []
    accs = []

    # weights
    w_out = float(lcfg.get("w_outcome", 1.0))
    w_gps = float(lcfg.get("w_gps", 1.0))
    w_tr = float(lcfg.get("w_target_reg", 0.0))
    w_mo = float(lcfg.get("w_mono", 0.0))

    mono_delta = float(lcfg.get("mono_delta", 0.05))
    mono_dir = str(lcfg.get("mono_direction", "decreasing"))
    eps_l2 = float(lcfg.get("epsilon_l2", 1.0))

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model_forward(model, batch)

        logits = out["logits"]

        # If GPS outputs exist, use targeted logits, else use raw logits
        has_gps = ("mu" in out) and ("sigma" in out)
        if has_gps:
            logits_t = targeted_regularized_logits(logits, batch["t"], out["mu"], out["sigma"], epsilon)
        else:
            logits_t = logits

        loss_y = outcome_bce_logits(batch["y"], logits_t)

        loss = w_out * loss_y

        # GPS loss (if available)
        if has_gps:
            loss_g = gps_gaussian_nll(batch["t"], out["mu"], out["sigma"])
            loss += w_gps * loss_g

            # epsilon penalty
            loss += w_tr * epsilon_penalty(epsilon, eps_l2)

            # monotonicity penalty (only if model uses t)
            if w_mo > 0:
                loss_m = monotonicity_penalty(model, batch, out, mono_delta, mono_dir)
                loss += w_mo * loss_m

        losses.append(loss.item())

        pred = (torch.sigmoid(logits) > 0.5).float()
        accs.append((pred == batch["y"]).float().mean().item())

    return float(np.mean(losses)), float(np.mean(accs))


# -----------------------------
# Main
# -----------------------------
def main():
    cfg = load_config()

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    dcfg = cfg["data"]
    tcfg = cfg["train"]
    lcfg = cfg.get("loss", {})

    # ---------------- train / valid data ----------------
    x_num_tr, x_cat_tr, t_tr, y_tr = generate_synthetic(
        n=dcfg["n_train"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"],
    )

    # validation (required for "best.pt")
    n_valid = int(dcfg.get("n_valid", max(10000, int(0.2 * dcfg["n_train"]))))
    x_num_va, x_cat_va, t_va, y_va = generate_synthetic(
        n=n_valid,
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"] + 1,
    )

    train_ds = CausalDataset(x_num_tr, x_cat_tr, t_tr, y_tr)
    valid_ds = CausalDataset(x_num_va, x_cat_va, t_va, y_va)

    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=tcfg["batch_size"], shuffle=False)

    # ---------------- model ----------------
    model = build_model(dcfg, cfg["model"], device)

    # optimizers
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=tcfg["lr"],
        weight_decay=tcfg.get("weight_decay", 0.0),
    )

    # targeted reg epsilon (learnable scalar) only used if model outputs GPS
    epsilon_init = float(lcfg.get("epsilon_init", 0.0))
    epsilon = torch.nn.Parameter(torch.tensor([epsilon_init], device=device))
    opt_eps = torch.optim.Adam([epsilon], lr=tcfg["lr"])

    best_val = float("inf")
    os.makedirs("artifacts", exist_ok=True)

    # ---------------- training loop ----------------
    for epoch in range(int(tcfg["epochs"])):
        model.train()
        losses = []

        # weights
        w_out = float(lcfg.get("w_outcome", 1.0))
        w_gps = float(lcfg.get("w_gps", 1.0))
        w_tr = float(lcfg.get("w_target_reg", 0.0))
        w_mo = float(lcfg.get("w_mono", 0.0))

        mono_delta = float(lcfg.get("mono_delta", 0.05))
        mono_dir = str(lcfg.get("mono_direction", "decreasing"))
        eps_l2 = float(lcfg.get("epsilon_l2", 1.0))

        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model_forward(model, batch)

            logits = out["logits"]

            has_gps = ("mu" in out) and ("sigma" in out)

            if has_gps:
                logits_t = targeted_regularized_logits(logits, batch["t"], out["mu"], out["sigma"], epsilon)
            else:
                logits_t = logits

            loss_y = outcome_bce_logits(batch["y"], logits_t)
            loss = w_out * loss_y

            # add GPS + targeted reg + monotonicity only if GPS exists
            if has_gps:
                loss_g = gps_gaussian_nll(batch["t"], out["mu"], out["sigma"])
                loss += w_gps * loss_g

                loss += w_tr * epsilon_penalty(epsilon, eps_l2)

                if w_mo > 0:
                    loss_m = monotonicity_penalty(model, batch, out, mono_delta, mono_dir)
                    loss += w_mo * loss_m

            optimizer.zero_grad(set_to_none=True)
            opt_eps.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg.get("grad_clip", 5.0)))
            optimizer.step()
            opt_eps.step()

            losses.append(loss.item())

        train_loss = float(np.mean(losses))
        val_loss, val_acc = evaluate(model, valid_loader, device, epsilon, lcfg)

        print(
            f"Epoch {epoch + 1} | TrainLoss: {train_loss:.4f} | "
            f"ValLoss: {val_loss:.4f} | ValAcc: {val_acc:.4f} | Eps: {epsilon.item():.4f}"
        )

        # save best checkpoint
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint_local(
                "artifacts/best.pt",
                {
                    "model_state": model.state_dict(),
                    "epsilon": epsilon.detach().cpu(),
                    "best_val_loss": best_val,
                    "config": cfg,
                },
            )
            print("Saved artifacts/best.pt")

    # also save last (useful)
    save_checkpoint_local(
        "artifacts/last.pt",
        {
            "model_state": model.state_dict(),
            "epsilon": epsilon.detach().cpu(),
            "config": cfg,
        },
    )
    print("Saved artifacts/last.pt")


if __name__ == "__main__":
    main()
