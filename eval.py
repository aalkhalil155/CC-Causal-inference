import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

from .utils import get_device, set_seed, batch_to_device
from .data import generate_synthetic, CausalDataset
from .preprocess import Standardizer
from .model import DragonNetContinuousAdvanced


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

    # preprocess (must match train)
    if pcfg.get("standardize_numeric", True):
        x_scaler = Standardizer().fit(x_num)  # NOTE: for demo we fit on valid itself
        # If you want strict correctness: load scalers from checkpoint and apply them.
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

    p = 1 / (1 + np.exp(-logit))
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
# 2) GPS overlap (diagnostics)
# -------------------------
def plot_gps_overlap(t, mu, sigma, outpath=None):
    """
    A practical overlap diagnostic:
    - compute standardized residual r = (t - mu)/sigma
    - if GPS is reasonable and overlap exists, residuals should be not-too-separated
    - also plot t distribution vs mu distribution
    """
    r = (t - mu) / (sigma + 1e-8)

    plt.figure()
    plt.hist(r, bins=60, density=True, alpha=0.8)
    plt.xlabel("Standardized residual r=(t-mu)/sigma")
    plt.ylabel("Density")
    plt.title("GPS Overlap Diagnostic: standardized residuals")
    plt.grid(True)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()

    # t vs mu hist overlay (not perfect “overlap”, but useful)
    plt.figure()
    plt.hist(t, bins=60, density=True, alpha=0.6, label="Observed t")
    plt.hist(mu, bins=60, density=True, alpha=0.6, label="Predicted mu(x)")
    plt.xlabel("t")
    plt.ylabel("Density")
    plt.title("Observed treatment vs predicted mean mu(x)")
    plt.legend()
    plt.grid(True)
    if outpath:
        base = outpath.replace(".png", "")
        plt.savefig(base + "_t_vs_mu.png", bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()

    # sigma sanity
    plt.figure()
    plt.hist(sigma, bins=60, density=True, alpha=0.8)
    plt.xlabel("Predicted sigma(x)")
    plt.ylabel("Density")
    plt.title("GPS sigma(x) distribution")
    plt.grid(True)
    if outpath:
        base = outpath.replace(".png", "")
        plt.savefig(base + "_sigma.png", bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# 3) Heterogeneity (dose-response by segments)
# -------------------------
@torch.no_grad()
def plot_heterogeneity_dose_response(
    model,
    x_num,
    x_cat,
    t_obs,
    device,
    n_bins=5,
    grid_points=40,
    outpath=None,
):
    """
    Segment customers by a simple proxy of "sensitivity":
    |d p / d t| at observed t (finite diff). Then plot dose-response curve by segment.
    """
    model.eval()

    # compute sensitivity proxy using finite difference at observed t
    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)
    t_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)

    delta = 0.05
    out0 = model(x_num_t, x_cat_t, t_t)
    out1 = model(x_num_t, x_cat_t, t_t + delta)

    p0 = torch.sigmoid(out0["logits"]).detach().cpu().numpy().reshape(-1)
    p1 = torch.sigmoid(out1["logits"]).detach().cpu().numpy().reshape(-1)

    sens = np.abs((p1 - p0) / delta)  # proxy |dp/dt|
    # bin into quantiles
    qs = np.quantile(sens, np.linspace(0, 1, n_bins + 1))
    seg = np.digitize(sens, qs[1:-1], right=True)  # 0..n_bins-1

    # treatment grid based on observed range
    t_min, t_max = np.quantile(t_obs, 0.01), np.quantile(t_obs, 0.99)
    t_grid = np.linspace(t_min, t_max, grid_points).astype(np.float32)

    plt.figure()
    for k in range(n_bins):
        idx = np.where(seg == k)[0]
        if len(idx) == 0:
            continue

        # compute mean predicted p across grid for this segment
        # (sample subset if too large for speed)
        if len(idx) > 8000:
            idx = np.random.choice(idx, size=8000, replace=False)

        x_num_k = x_num_t[idx]
        x_cat_k = x_cat_t[idx]

        ps = []
        for tg in t_grid:
            t_k = torch.full((x_num_k.shape[0], 1), tg, device=device)
            outg = model(x_num_k, x_cat_k, t_k)
            pg = torch.sigmoid(outg["logits"]).mean().item()
            ps.append(pg)

        plt.plot(t_grid, ps, label=f"Segment {k+1} (sens quantile)")

    plt.xlabel("Treatment t (standardized)")
    plt.ylabel("Mean predicted P(Y=1 | x, t)")
    plt.title("Heterogeneity: dose-response by sensitivity segments")
    plt.grid(True)
    plt.legend()

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

    # treatment grid based on standardized convention (centered-ish)
    # For synthetic it’s fine; for real use quantiles of observed t.
    t_grid = np.linspace(-2.0, 2.0, grid_points).astype(np.float32)

    # compute p for each grid point in batches
    N = x_num_t.shape[0]
    batch = 4096

    P = np.zeros((N, grid_points), dtype=np.float32)

    for j, tg in enumerate(t_grid):
        t_col = torch.full((N, 1), tg, device=device)
        # chunk for memory
        ps = []
        for i in range(0, N, batch):
            out = model(x_num_t[i:i+batch], x_cat_t[i:i+batch], t_col[i:i+batch])
            ps.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        P[:, j] = np.vstack(ps).reshape(-1)

    # violations
    d = np.diff(P, axis=1)  # P(t_{j+1}) - P(t_j)
    if direction == "decreasing":
        viol = d > 1e-6
    else:
        viol = d < -1e-6

    viol_rate = viol.mean(axis=1)

    # plot histogram of violation rates
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

    # plot average curve
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

    # GPS overlap
    plot_gps_overlap(pred["t"], pred["mu"], pred["sigma"], outpath=f"{args.outdir}/gps_overlap.png")
    print("Saved GPS overlap plots.")

    # Heterogeneity dose-response
    plot_heterogeneity_dose_response(
        model,
        pred["x_num"],
        pred["x_cat"],
        pred["t"],
        device,
        n_bins=5,
        grid_points=40,
        outpath=f"{args.outdir}/heterogeneity_dose_response.png",
    )
    print("Saved heterogeneity plot.")

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
