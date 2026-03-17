import argparse
import yaml
import numpy as np
import torch
import os
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import spearmanr, gaussian_kde

from utils import get_device, set_seed, batch_to_device
from data import generate_synthetic, CausalDataset
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


def transform_with_saved_scalers(x_num, t, ckpt):
    x_mean = ckpt.get("x_scaler_mean", None)
    x_std = ckpt.get("x_scaler_std", None)
    t_mean = ckpt.get("t_scaler_mean", None)
    t_std = ckpt.get("t_scaler_std", None)

    if x_mean is not None and x_std is not None:
        x_num = (x_num - np.asarray(x_mean)) / np.asarray(x_std)

    if t_mean is not None and t_std is not None:
        t = ((t.reshape(-1, 1) - np.asarray(t_mean)) / np.asarray(t_std)).reshape(-1)

    return x_num.astype(np.float32), t.astype(np.float32)


def make_valid_loader(cfg: dict, ckpt: dict, seed_offset: int = 1):
    dcfg = cfg["data"]

    x_num, x_cat, t, y = generate_synthetic(
        n=dcfg["n_valid"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"] + seed_offset,
    )

    x_num, t = transform_with_saved_scalers(x_num, t, ckpt)

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
# ROC
# -------------------------
def plot_roc(y, p, outpath=None):
    auc = roc_auc_score(y, p)
    fpr, tpr, _ = roc_curve(y, p)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, linewidth=2, label=f"OOT AUC={auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()
    return auc


# -------------------------
# GPS overlap (smoothed)
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
    gps = gaussian_pdf(t, mu, sigma)
    gps = np.clip(gps, np.quantile(gps, 0.001), np.quantile(gps, 0.999))

    qs = np.quantile(t, np.linspace(0, 1, n_quantiles + 1))
    q_idx = np.digitize(t, qs[1:-1], right=True)

    plt.figure(figsize=(8, 5))

    x_grid = np.linspace(gps.min(), gps.max(), 300)
    for k in range(n_quantiles):
        vals = gps[q_idx == k]
        if len(vals) < 20:
            continue

        kde = gaussian_kde(vals)
        y_grid = kde(x_grid)
        plt.plot(x_grid, y_grid, linewidth=2, label=f"Q{k+1}")

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
# Heterogeneity
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
    model.eval()

    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)

    N = x_num_t.shape[0]
    p_min = np.full(N, np.inf, dtype=np.float32)
    p_max = np.full(N, -np.inf, dtype=np.float32)

    for tg in t_grid.astype(np.float32):
        for i in range(0, N, batch_size):
            xb = x_num_t[i:i + batch_size]
            cb = x_cat_t[i:i + batch_size]
            tb = torch.full((xb.shape[0], 1), float(tg), device=device)

            out = model(xb, cb, tb)
            p = torch.sigmoid(out["logits"]).detach().cpu().numpy().reshape(-1)

            p_min[i:i + batch_size] = np.minimum(p_min[i:i + batch_size], p)
            p_max[i:i + batch_size] = np.maximum(p_max[i:i + batch_size], p)

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

    v = float(np.mean(H)) if vline == "mean" else float(np.median(H))

    plt.figure(figsize=(8, 5))
    plt.hist(H, bins=bins, density=False, alpha=0.55)

    x_grid = np.linspace(H.min(), H.max(), 300)
    kde = gaussian_kde(H)
    y_grid = kde(x_grid)
    scale = len(H) * (H.max() - H.min()) / bins
    plt.plot(x_grid, y_grid * scale, linewidth=2)

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
# Monotonicity
# -------------------------
@torch.no_grad()
def compute_local_treatment_effects(
    model,
    x_num,
    x_cat,
    t_obs,
    device,
    delta=0.02,
    batch_size=4096,
):
    model.eval()

    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)
    t_t = torch.tensor(t_obs, dtype=torch.float32, device=device).view(-1, 1)

    N = x_num_t.shape[0]
    d_hat = np.zeros(N, dtype=np.float32)

    for i in range(0, N, batch_size):
        xb = x_num_t[i:i + batch_size]
        cb = x_cat_t[i:i + batch_size]
        tb = t_t[i:i + batch_size]

        out_plus = model(xb, cb, tb + delta)
        out_minus = model(xb, cb, tb - delta)

        p_plus = torch.sigmoid(out_plus["logits"]).detach().cpu().numpy().reshape(-1)
        p_minus = torch.sigmoid(out_minus["logits"]).detach().cpu().numpy().reshape(-1)

        d_hat[i:i + batch_size] = (p_plus - p_minus) / (2.0 * delta)

    return np.asarray(t_obs).reshape(-1), d_hat


def plot_monotonicity_scatter(
    t_obs,
    d_hat,
    bins=40,
    sample_points=12000,
    title_prefix="Monotonicity",
    outpath=None,
):
    mask = np.isfinite(t_obs) & np.isfinite(d_hat)
    t_obs = t_obs[mask]
    d_hat = d_hat[mask]

    if len(t_obs) > sample_points:
        idx = np.random.choice(len(t_obs), size=sample_points, replace=False)
        t_scatter = t_obs[idx]
        d_scatter = d_hat[idx]
    else:
        t_scatter = t_obs
        d_scatter = d_hat

    rho, pval = spearmanr(t_obs, d_hat)

    edges = np.quantile(t_obs, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(t_obs.min(), t_obs.max(), bins + 1)

    centers, means = [], []
    for j in range(len(edges) - 1):
        if j < len(edges) - 2:
            m = (t_obs >= edges[j]) & (t_obs < edges[j + 1])
        else:
            m = (t_obs >= edges[j]) & (t_obs <= edges[j + 1])

        if m.sum() == 0:
            continue
        centers.append(0.5 * (edges[j] + edges[j + 1]))
        means.append(d_hat[m].mean())

    plt.figure(figsize=(8, 6))
    plt.scatter(t_scatter, d_scatter, s=8, alpha=0.2)
    plt.plot(centers, means, linewidth=3)
    plt.axhline(0.0, linestyle="--", linewidth=1.5, color="gray")
    plt.xlabel("Treatment")
    plt.ylabel("Local treatment effect dP/dt")
    plt.title(f"{title_prefix}\nSpearman ρ = {rho:.3f}, p = {pval:.3g}")
    plt.grid(True)

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


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
            out = model(x_num_t[i:i + batch], x_cat_t[i:i + batch], t_col[i:i + batch])
            ps.append(torch.sigmoid(out["logits"]).detach().cpu().numpy())
        P[:, j] = np.vstack(ps).reshape(-1)

    d = np.diff(P, axis=1)
    viol = (d > 1e-6) if direction == "decreasing" else (d < -1e-6)
    viol_rate = viol.mean(axis=1)

    plt.figure(figsize=(7, 5))
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

    plt.figure(figsize=(7, 5))
    plt.plot(t_grid, P.mean(axis=0), linewidth=2)
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
# Qini helpers: plug-in style
# -------------------------
@torch.no_grad()
def score_candidate_arms_prob(
    model,
    x_num,
    x_cat,
    device,
    candidate_arms,
    batch_size=4096,
):
    model.eval()

    x_num_t = torch.tensor(x_num, dtype=torch.float32, device=device)
    x_cat_t = torch.tensor(x_cat, dtype=torch.long, device=device)

    N = x_num_t.shape[0]
    K = len(candidate_arms)
    P = np.zeros((N, K), dtype=np.float32)

    for k, arm in enumerate(candidate_arms):
        for i in range(0, N, batch_size):
            xb = x_num_t[i:i + batch_size]
            cb = x_cat_t[i:i + batch_size]
            tb = torch.full((xb.shape[0], 1), float(arm), device=device)

            out = model(xb, cb, tb)
            p = torch.sigmoid(out["logits"]).detach().cpu().numpy().reshape(-1)
            P[i:i + batch_size, k] = p

    return P


def build_demo_spend_gain_from_prob(P, candidate_arms, balance):
    """
    Replace this later with your real mortgage formulas.
    """
    candidate_arms = np.asarray(candidate_arms, dtype=np.float32)
    balance = np.asarray(balance, dtype=np.float32).reshape(-1)
    arm_row = candidate_arms.reshape(1, -1)
    bal_col = balance.reshape(-1, 1)

    spend = np.maximum(arm_row, 0.0) * bal_col
    gain = P * arm_row * bal_col
    return spend, gain


def build_single_arm_curves(spend, gain, arm_labels=None):
    N, K = gain.shape
    curves = []

    for k in range(K):
        s = spend[:, k].copy()
        g = gain[:, k].copy()
        eff = np.where(s > 1e-12, g / s, -np.inf)
        order = np.argsort(-eff)

        cum_spend = np.cumsum(s[order])
        cum_gain = np.cumsum(g[order])

        label = f"Arm {k}" if arm_labels is None else arm_labels[k]
        curves.append({
            "label": label,
            "cum_spend": cum_spend,
            "cum_gain": cum_gain,
            "order": order,
        })

    return curves


def build_multi_arm_optimal_curve(spend, gain):
    eff = np.where(spend > 1e-12, gain / spend, -np.inf)
    best_idx = np.argmax(eff, axis=1)

    row_idx = np.arange(spend.shape[0])
    best_spend = spend[row_idx, best_idx]
    best_gain = gain[row_idx, best_idx]
    best_eff = eff[row_idx, best_idx]

    order = np.argsort(-best_eff)
    cum_spend = np.cumsum(best_spend[order])
    cum_gain = np.cumsum(best_gain[order])

    return {
        "best_idx": best_idx,
        "best_spend": best_spend,
        "best_gain": best_gain,
        "best_eff": best_eff,
        "cum_spend": cum_spend,
        "cum_gain": cum_gain,
        "order": order,
    }


def build_historical_curve(t_obs, candidate_arms, balance, p_obs):
    candidate_arms = np.asarray(candidate_arms, dtype=np.float32)
    t_obs = np.asarray(t_obs, dtype=np.float32).reshape(-1)
    balance = np.asarray(balance, dtype=np.float32).reshape(-1)
    p_obs = np.asarray(p_obs, dtype=np.float32).reshape(-1)

    nearest_idx = np.argmin(
        np.abs(t_obs.reshape(-1, 1) - candidate_arms.reshape(1, -1)),
        axis=1,
    )
    chosen_arm = candidate_arms[nearest_idx]

    spend_hist = np.maximum(chosen_arm, 0.0) * balance
    gain_hist = p_obs * chosen_arm * balance

    eff_hist = np.where(spend_hist > 1e-12, gain_hist / spend_hist, -np.inf)
    order = np.argsort(-eff_hist)

    cum_spend = np.cumsum(spend_hist[order])
    cum_gain = np.cumsum(gain_hist[order])

    return {
        "nearest_idx": nearest_idx,
        "chosen_arm": chosen_arm,
        "cum_spend": cum_spend,
        "cum_gain": cum_gain,
        "order": order,
    }


def bootstrap_multi_arm_ci(spend, gain, n_boot=50, seed=42):
    rng = np.random.default_rng(seed)
    N = spend.shape[0]

    base = build_multi_arm_optimal_curve(spend, gain)
    x_grid = base["cum_spend"]

    ys = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        spend_b = spend[idx]
        gain_b = gain[idx]

        curve_b = build_multi_arm_optimal_curve(spend_b, gain_b)
        xb = np.maximum.accumulate(curve_b["cum_spend"])
        yb = curve_b["cum_gain"]

        if xb[-1] <= 0:
            ys.append(np.zeros_like(x_grid))
            continue

        x_target = np.clip(x_grid, xb.min(), xb.max())
        ys.append(np.interp(x_target, xb, yb))

    ys = np.vstack(ys)
    return x_grid, np.percentile(ys, 2.5, axis=0), np.percentile(ys, 97.5, axis=0)


def plot_multi_arm_qini_curves(
    single_arm_curves,
    multi_arm_curve,
    historical_curve=None,
    ci_band=None,
    title="Qini Curves (Pointer Logic)",
    subtitle=None,
    outpath=None,
):
    plt.figure(figsize=(10, 6))

    for c in single_arm_curves:
        plt.plot(c["cum_spend"], c["cum_gain"], linestyle="--", linewidth=2, label=c["label"])

    plt.plot(
        multi_arm_curve["cum_spend"],
        multi_arm_curve["cum_gain"],
        linewidth=3.5,
        label="Multi-Arm (Optimal)",
    )

    if ci_band is not None:
        x_ci, y_lo, y_hi = ci_band
        plt.fill_between(x_ci, y_lo, y_hi, alpha=0.2, label="Multi-Arm 95% CI")

    if historical_curve is not None:
        plt.plot(
            historical_curve["cum_spend"],
            historical_curve["cum_gain"],
            linestyle=":",
            linewidth=3,
            label="Observed Policy (Historical)",
        )

    ttl = title if subtitle is None else f"{title}\n{subtitle}"
    plt.title(ttl)
    plt.xlabel("Cumulative Discount Spend ($)")
    plt.ylabel("Cumulative Gain (expected $ interest)")
    plt.grid(True)
    plt.legend()

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


def plot_best_arm_distribution(best_idx, candidate_arms, outpath=None):
    counts = np.bincount(best_idx, minlength=len(candidate_arms))

    plt.figure(figsize=(8, 5))
    plt.bar([str(a) for a in candidate_arms], counts)
    plt.xlabel("Chosen arm")
    plt.ylabel("Count")
    plt.title("Best chosen arm distribution")
    plt.grid(True, axis="y")

    if outpath:
        plt.savefig(outpath, bbox_inches="tight", dpi=150)
    else:
        plt.show()
    plt.close()


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--ckpt", default="artifacts/best.pt")
    ap.add_argument("--outdir", default="artifacts/plots")
    ap.add_argument("--gps_quantiles", type=int, default=5)
    ap.add_argument("--hetero_grid_points", type=int, default=60)
    ap.add_argument("--mono_delta", type=float, default=0.02)
    ap.add_argument("--n_boot", type=int, default=50)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg.get("device", "cpu"))
    os.makedirs(args.outdir, exist_ok=True)

    ckpt = load_checkpoint(args.ckpt, device)

    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)

    loader = make_valid_loader(cfg, ckpt)
    pred = predict_all(model, loader, device)

    # ROC
    auc = plot_roc(pred["y"], pred["p"], outpath=f"{args.outdir}/roc.png")
    print(f"AUC: {auc:.4f}")

    # GPS overlap
    plot_gps_overlap_by_treatment_quantiles(
        pred["t"],
        pred["mu"],
        pred["sigma"],
        n_quantiles=args.gps_quantiles,
        title="Overlap Plot for OOT",
        outpath=f"{args.outdir}/gps_overlap_quantiles.png",
    )
    print("Saved GPS overlap plot.")

    # Heterogeneity
    t_min, t_max = np.quantile(pred["t"], 0.01), np.quantile(pred["t"], 0.99)
    t_grid = np.linspace(t_min, t_max, args.hetero_grid_points).astype(np.float32)

    H, _, _ = compute_heterogeneity_max_min(
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
    t_eff, d_eff = compute_local_treatment_effects(
        model=model,
        x_num=pred["x_num"],
        x_cat=pred["x_cat"],
        t_obs=pred["t"],
        device=device,
        delta=args.mono_delta,
        batch_size=4096,
    )

    plot_monotonicity_scatter(
        t_obs=t_eff,
        d_hat=d_eff,
        bins=40,
        sample_points=12000,
        title_prefix="Monotonicity",
        outpath=f"{args.outdir}/monotonicity_scatter.png",
    )
    print("Saved monotonicity scatter plot.")

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

    # Qini-style curves
    candidate_arms = np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14], dtype=np.float32)
    arm_labels = [f"Single-Arm: Bin {i+1}" for i in range(len(candidate_arms))]

    # Demo proxy only. Replace with your real balance later.
    fake_balance = (50000.0 * np.exp(0.25 * pred["x_num"][:, 0])).astype(np.float32)

    P_arms = score_candidate_arms_prob(
        model=model,
        x_num=pred["x_num"],
        x_cat=pred["x_cat"],
        device=device,
        candidate_arms=candidate_arms,
        batch_size=4096,
    )

    spend, gain = build_demo_spend_gain_from_prob(
        P=P_arms,
        candidate_arms=candidate_arms,
        balance=fake_balance,
    )

    single_arm_curves = build_single_arm_curves(
        spend=spend,
        gain=gain,
        arm_labels=arm_labels,
    )

    multi_arm_curve = build_multi_arm_optimal_curve(
        spend=spend,
        gain=gain,
    )

    historical_curve = build_historical_curve(
        t_obs=pred["t"],
        candidate_arms=candidate_arms,
        balance=fake_balance,
        p_obs=pred["p"],
    )

    ci_band = bootstrap_multi_arm_ci(
        spend=spend,
        gain=gain,
        n_boot=args.n_boot,
        seed=cfg["seed"],
    )

    plot_multi_arm_qini_curves(
        single_arm_curves=single_arm_curves,
        multi_arm_curve=multi_arm_curve,
        historical_curve=historical_curve,
        ci_band=ci_band,
        title="Qini Curves (Pointer Logic)",
        subtitle="(Demo Synthetic Data)",
        outpath=f"{args.outdir}/multi_arm_qini_curves.png",
    )

    plot_best_arm_distribution(
        best_idx=multi_arm_curve["best_idx"],
        candidate_arms=candidate_arms,
        outpath=f"{args.outdir}/multi_arm_best_arm_distribution.png",
    )

    print("Saved multi-arm Qini-style curves.")
    print(f"All plots saved to: {args.outdir}")


if __name__ == "__main__":
    main()