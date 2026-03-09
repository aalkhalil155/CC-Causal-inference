import argparse
import yaml
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from utils import get_device, set_seed, batch_to_device
from data import generate_synthetic, CausalDataset
from preprocess import Standardizer
from model import DragonNetContinuousAdvanced


# -------------------------
# Helpers
# -------------------------
@torch.no_grad()
def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    return ckpt


def build_model_from_config(cfg: dict) -> DragonNetContinuousAdvanced:
    dcfg = cfg["data"]
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
    )
    return model


def make_valid_loader(cfg: dict, seed_offset: int = 1):
    # Regenerate the same style of validation set as train.py
    dcfg = cfg["data"]
    pcfg = cfg.get("preprocess", {})

    x_num, x_cat, t, y = generate_synthetic(
        n=dcfg["n_valid"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"] + seed_offset,
    )

    # preprocess (must match train as closely as possible)
    if pcfg.get("standardize_numeric", True):
        x_scaler = Standardizer().fit(x_num)
        x_num = x_scaler.transform(x_num).astype(np.float32)

    if pcfg.get("standardize_treatment", True):
        t_scaler = Standardizer().fit(t.reshape(-1, 1))
        t = t_scaler.transform(t.reshape(-1, 1)).reshape(-1).astype(np.float32)

    ds = CausalDataset(x_num, x_cat, t, y)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=0)
    return loader


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    xs_num, xs_cat, ts, ys = [], [], [], []
    logits, mus, sigmas, zs = [], [], [], []

    for batch in loader:
        batch = batch_to_device(batch, device)
        out = model(batch["x_num"], batch["x_cat"], batch["t"])

        xs_num.append(batch["x_num"].cpu())
        xs_cat.append(batch["x_cat"].cpu())
        ts.append(batch["t"].cpu())
        ys.append(batch["y"].cpu())

        logits.append(out["logits"].cpu())
        mus.append(out["mu"].cpu())
        sigmas.append(out["sigma"].cpu())
        zs.append(out["z"].cpu())

    x_num = torch.cat(xs_num).numpy()
    x_cat = torch.cat(xs_cat).numpy()
    t = torch.cat(ts).numpy().reshape(-1)
    y = torch.cat(ys).numpy().reshape(-1)
    logit = torch.cat(logits).numpy().reshape(-1)
    mu = torch.cat(mus).numpy().reshape(-1)
    sigma = torch.cat(sigmas).numpy().reshape(-1)
    z = torch.cat(zs).numpy()

    p = 1.0 / (1.0 + np.exp(-logit))
    return dict(x_num=x_num, x_cat=x_cat, t=t, y=y, logit=logit, p=p, mu=mu, sigma=sigma, z=z)


# -------------------------
# 1) ROC / AUC
# -------------------------
def plot_roc(y, p, outpath=None):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC={auc:.4f})")
    plt.grid(True)

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()
    return auc


# -------------------------
# 2) GPS overlap (NEW: matches your screenshot style)
# -------------------------
def gaussian_pdf(t, mu, sigma):
    sigma = np.clip(sigma, 1e-6, None)
    z = (t - mu) / sigma
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * z * z)


def plot_gps_overlap_by_treatment_quantiles(
    t,
    mu,
    sigma,
    n_quantiles=5,
    title="Overlap Plot for OOT",
    outpath=None,
):
    """
    GPS overlap plot:
    1) compute GPS score r(t,x)=f(t|x) using Gaussian(mu(x), sigma(x))
    2) split samples by treatment quantiles Q1..QK
    3) plot density curves of GPS score for each treatment quantile
    """
    gps = gaussian_pdf(t, mu, sigma)

    # treatment quantiles
    qs = np.quantile(t, np.linspace(0, 1, n_quantiles + 1))
    q_idx = np.digitize(t, qs[1:-1], right=True)  # 0..K-1

    plt.figure(figsize=(8, 5))

    for k in range(n_quantiles):
        mask = (q_idx == k)
        if mask.sum() < 20:
            continue

        vals = gps[mask]
        hist, edges = np.histogram(vals, bins=80, density=True)
        centers = 0.5 * (edges[1:] + edges[:-1])
        plt.plot(centers, hist, label=f"Q{k+1}")

    plt.xlabel("GPS")
    plt.ylabel("Density")
    plt.title(title)
    plt.grid(True)
    plt.legend(title="treatment_quantile")

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# 3) Heterogeneity (NEW: max-min across t grid per loan)
# -------------------------
@torch.no_grad()
def compute_heterogeneity_max_min(
    model,
    x_num,
    x_cat,
    device,
    t_grid,
    batch_size=4096,
):
    """
    For each loan i:
      H_i = max_t p_i(t) - min_t p_i(t)

    Returns:
      H: (N,)
      p_min: (N,)
      p_max: (N,)
    """
    model.eval()

    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)

    N = x_num_t.shape[0]
    p_min = np.full(N, np.inf, dtype=np.float32)
    p_max = np.full(N, -np.inf, dtype=np.float32)

    for tg in t_grid.astype(np.float32):
        for i in range(0, N, batch_size):
            xb = x_num_t[i:i+batch_size]
            cb = x_cat_t[i:i+batch_size]
            tb = torch.full((xb.shape[0], 1), float(tg), device=device)

            out = model(xb, cb, tb)
            p = torch.sigmoid(out["logits"]).detach().cpu().numpy().reshape(-1)

            p_min[i:i+batch_size] = np.minimum(p_min[i:i+batch_size], p)
            p_max[i:i+batch_size] = np.maximum(p_max[i:i+batch_size], p)

    H = (p_max - p_min).astype(np.float32)
    return H, p_min, p_max


def plot_heterogeneity_max_min_hist(
    H,
    title="Distribution of Model-Discount Sensitivity Across Loans",
    xlabel="Change in Acceptance Probability (Max - Min)",
    bins=40,
    vline="mean",
    outpath=None,
):
    H = np.asarray(H).reshape(-1)
    H = H[np.isfinite(H)]

    if vline == "median":
        v = float(np.median(H))
    else:
        v = float(np.mean(H))

    plt.figure(figsize=(8, 5))
    plt.hist(H, bins=bins, density=False, alpha=0.6)

    hist, edges = np.histogram(H, bins=120, density=False)
    centers = 0.5 * (edges[1:] + edges[:-1])
    plt.plot(centers, hist)

    plt.axvline(v, linestyle="--", linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.grid(True)

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# 4) Monotonicity diagnostics
# -------------------------
@torch.no_grad()
def plot_monotonicity_violations(
    model,
    x_num,
    x_cat,
    device,
    direction="decreasing",
    grid_points=50,
    outpath=None,
):
    """
    For each customer, evaluate p(t) on a grid and compute:
      violation_rate = fraction of adjacent steps violating monotonicity
    Plot histogram of violation rates + average curve.
    """
    model.eval()
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)

    t_grid = np.linspace(-2.0, 2.0, grid_points).astype(np.float32)

    N = x_num_t.shape[0]
    batch = 4096
    P = np.zeros((N, grid_points), dtype=np.float32)

    for j, tg in enumerate(t_grid):
        t_col = torch.full((N, 1), tg, device=device)
        ps = []
        for i in range(0, N, batch):
            out = model(x_num_t[i:i+batch], x_cat_t[i:i+batch], t_col[i:i+batch])
            ps.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        P[:, j] = np.vstack(ps).reshape(-1)

    d = np.diff(P, axis=1)
    if direction == "decreasing":
        viol = d > 1e-6
    else:
        viol = d < -1e-6

    viol_rate = viol.mean(axis=1)

    plt.figure()
    plt.hist(viol_rate, bins=50, density=True, alpha=0.8)
    plt.xlabel("Per-customer monotonicity violation rate")
    plt.ylabel("Density")
    plt.title(f"Monotonicity violations ({direction})")
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()

    plt.figure()
    plt.plot(t_grid, P.mean(axis=0))
    plt.xlabel("Treatment t (standardized)")
    plt.ylabel("Mean predicted P(Y=1 | x, t)")
    plt.title("Average dose-response curve (sanity check)")
    plt.grid(True)
    if outpath:
        base = outpath.replace(".png", "")
        plt.savefig(base + "_avg_curve.png", bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="artifacts/best.pt")
    ap.add_argument("--outdir", default="artifacts/plots")
    ap.add_argument("--gps_quantiles", type=int, default=5)
    ap.add_argument("--hetero_grid_points", type=int, default=60)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))

    os.makedirs(args.outdir, exist_ok=True)

    # build model + load weights
    model = build_model_from_config(cfg).to(device)
    ckpt = load_checkpoint(args.ckpt, device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    # get validation data predictions
    loader = make_valid_loader(cfg)
    pred = predict_all(model, loader, device)

    # ROC
    auc = plot_roc(pred["y"], pred["p"], outpath=f"{args.outdir}/roc.png")
    print(f"AUC: {auc:.4f}")

    # GPS overlap (new)
    plot_gps_overlap_by_treatment_quantiles(
        pred["t"],
        pred["mu"],
        pred["sigma"],
        n_quantiles=args.gps_quantiles,
        title="Overlap Plot for OOT",
        outpath=f"{args.outdir}/gps_overlap_quantiles.png",
    )
    print("Saved GPS overlap plot.")

    # Heterogeneity (new: max-min over treatment grid)
    t_min, t_max = np.quantile(pred["t"], 0.01), np.quantile(pred["t"], 0.99)
    t_grid = np.linspace(t_min, t_max, args.hetero_grid_points).astype(np.float32)

    H, p_min, p_max = compute_heterogeneity_max_min(
        model=model,
        x_num=pred["x_num"],
        x_cat=pred["x_cat"],
        device=device,
        t_grid=t_grid,
        batch_size=4096,
    )

    plot_heterogeneity_max_min_hist(
        H,
        title="Distribution of Model-Discount Sensitivity Across Loans",
        xlabel="Change in Acceptance Probability (Max - Min)",
        bins=40,
        vline="mean",
        outpath=f"{args.outdir}/heterogeneity_max_min.png",
    )
    print("Saved heterogeneity max-min plot.")

    # Monotonicity
    plot_monotonicity_violations(
        model,
        pred["x_num"],
        pred["x_cat"],
        device,
        direction=cfg["loss"].get("mono_direction", "decreasing"),
        grid_points=50,
        outpath=f"{args.outdir}/monotonicity.png",
    )
    print("Saved monotonicity plots.")

    print(f"All plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()