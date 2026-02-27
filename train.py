import os
import yaml
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

    return float(np.mean(losses)), float(np.mean(accs))

def main(config_path: str = "configs/config.yaml"):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

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
    x_num_va, x_cat_va, t_va, y_va = generate_synthetic(
        n=dcfg["n_valid"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"] + 1,
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
    valid_ds = CausalDataset(x_num_va, x_cat_va, t_va, y_va)

    tcfg = cfg["train"]
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=tcfg["batch_size"], shuffle=False, num_workers=0)

    mcfg = cfg["model"]
    model = DragonNetContinuousAdvanced(
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        emb_dim=mcfg["emb_dim"],
        d_hidden=mcfg["d_hidden"],
        n_shared_layers=mcfg["n_shared_layers"],
        dropout=mcfg["dropout"],
        min_sigma=mcfg["min_sigma"],
        use_t_mlp=mcfg.get("use_t_mlp", True),
        t_mlp_hidden=mcfg.get("t_mlp_hidden", 64),
        t_mlp_layers=mcfg.get("t_mlp_layers", 2),
        use_gamma=mcfg.get("use_gamma", False),
        use_cat_offset=mcfg.get("use_cat_offset", False),
        cat_offset_hidden=mcfg.get("cat_offset_hidden", 64),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))

    lcfg = cfg["loss"]
    epsilon = torch.nn.Parameter(torch.tensor([float(lcfg.get("epsilon_init", 0.0))], device=device))
    opt_eps = torch.optim.Adam([epsilon], lr=tcfg["lr"])

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

        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint("artifacts/best.pt", {
                "model_state": model.state_dict(),
                "epsilon": epsilon.detach().cpu(),
                "x_scaler_mean": None if x_scaler is None else x_scaler.mean_,
                "x_scaler_std": None if x_scaler is None else x_scaler.std_,
                "t_scaler_mean": None if t_scaler is None else t_scaler.mean_,
                "t_scaler_std": None if t_scaler is None else t_scaler.std_,
                "config": cfg,
            })
            print("Saved best checkpoint.\n")

if __name__ == "__main__":
    main()
