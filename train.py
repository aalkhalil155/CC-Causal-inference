import pathlib
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import set_seed, get_device, batch_to_device, save_checkpoint
from .data import generate_synthetic, CausalDataset
from .preprocess import Standardizer
from .model import DragonNetContinuousAdvanced
from .losses import (
    outcome_bce_logits,
    gps_gaussian_nll,
    targeted_regularized_logits,
    epsilon_penalty,
    monotonicity_penalty,
)

@torch.no_grad()
def evaluate(model, loader, device, epsilon):
    model.eval()
    losses = []
    accs = []
    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch["x_num"], batch["x_cat"], batch["t"])

        logits_t = targeted_regularized_logits(out["logits"], batch["t"], out["mu"], out["sigma"], epsilon)

        loss_y = outcome_bce_logits(batch["y"], logits_t)
        loss_g = gps_gaussian_nll(batch["t"], out["mu"], out["sigma"])
        losses.append((loss_y + loss_g).item())

        pred = (torch.sigmoid(out["logits"]) > 0.5).float()
        accs.append((pred == batch["y"]).float().mean().item())

# Support both execution modes:
# 1) `python train.py` (script mode)
# 2) `python -m <package>.train` (package mode)
if __package__ in (None, ""):
    repo_root = pathlib.Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from data import CausalDataset, generate_synthetic
    from model import DragonNetContinuous
    from utils import get_device, set_seed
else:
    from .data import CausalDataset, generate_synthetic
    from .model import DragonNetContinuous
    from .utils import get_device, set_seed


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


def main():
    cfg = load_config()

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))

    dcfg = cfg["data"]
    x_num_tr, x_cat_tr, t_tr, y_tr = generate_synthetic(
        n=dcfg["n_train"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"],
    )

    pcfg = cfg.get("preprocess", {})
    x_scaler = Standardizer().fit(x_num_tr) if pcfg.get("standardize_numeric", True) else None
    t_scaler = Standardizer().fit(t_tr.reshape(-1, 1)) if pcfg.get("standardize_treatment", True) else None

    if x_scaler is not None:
        x_num_tr = x_scaler.transform(x_num_tr).astype(np.float32)
        x_num_va = x_scaler.transform(x_num_va).astype(np.float32)

    if t_scaler is not None:
        t_tr = t_scaler.transform(t_tr.reshape(-1, 1)).reshape(-1).astype(np.float32)
        t_va = t_scaler.transform(t_va.reshape(-1, 1)).reshape(-1).astype(np.float32)

    train_ds = CausalDataset(x_num_tr, x_cat_tr, t_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )

    mcfg = cfg["model"]
    model = DragonNetContinuousAdvanced(
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        emb_dim=cfg["model"]["emb_dim"],
        d_hidden=cfg["model"]["d_hidden"],
        n_shared_layers=cfg["model"]["n_shared_layers"],
        dropout=cfg["model"]["dropout"],
        min_sigma=cfg["model"]["min_sigma"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    bce = torch.nn.BCEWithLogitsLoss()

    best_val = float("inf")
    os.makedirs("artifacts", exist_ok=True)

    step = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{tcfg['epochs']}")
        running = []

        for batch in pbar:
            step += 1
            batch = batch_to_device(batch, device)
            out = model(batch["x_num"], batch["x_cat"], batch["t"])

            logits_t = targeted_regularized_logits(out["logits"], batch["t"], out["mu"], out["sigma"], epsilon)

            loss_y = outcome_bce_logits(batch["y"], logits_t)
            loss_g = gps_gaussian_nll(batch["t"], out["mu"], out["sigma"])
            loss_tr = epsilon_penalty(epsilon, epsilon_l2=float(lcfg.get("epsilon_l2", 1.0)))
            loss_m = monotonicity_penalty(
                model,
                batch["x_num"],
                batch["x_cat"],
                batch["t"],
                delta=float(lcfg.get("mono_delta", 0.05)),
                direction=str(lcfg.get("mono_direction", "decreasing")),
            )

            loss = (
                float(lcfg.get("w_outcome", 1.0)) * loss_y
                + float(lcfg.get("w_gps", 1.0)) * loss_g
                + float(lcfg.get("w_target_reg", 0.0)) * loss_tr
                + float(lcfg.get("w_mono", 0.0)) * loss_m
            )

            opt.zero_grad(set_to_none=True)
            opt_eps.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            if tcfg.get("grad_clip", None) is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))

            opt.step()
            opt_eps.step()

            running.append(loss.item())
            if step % int(tcfg.get("log_every_steps", 50)) == 0:
                pbar.set_postfix({
                    "loss": f"{np.mean(running[-100:]):.4f}",
                    "bce": f"{loss_y.item():.4f}",
                    "gps": f"{loss_g.item():.4f}",
                    "mono": f"{loss_m.item():.4f}",
                    "eps": f"{epsilon.item():.4f}",
                })

        val_loss, val_acc = evaluate(model, valid_loader, device, epsilon)
        print(f"\nVal: loss={val_loss:.4f}, acc={val_acc:.4f}, epsilon={epsilon.item():.4f}")

        print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
