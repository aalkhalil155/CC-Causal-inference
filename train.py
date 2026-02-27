import pathlib
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

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
    device = get_device(cfg["device"])

    dcfg = cfg["data"]

    x_num_tr, x_cat_tr, t_tr, y_tr = generate_synthetic(
        n=dcfg["n_train"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        treatment_noise=dcfg["treatment_noise"],
        outcome_noise=dcfg["outcome_noise"],
        seed=cfg["seed"],
    )

    train_ds = CausalDataset(x_num_tr, x_cat_tr, t_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )

    model = DragonNetContinuous(
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

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["x_num"], batch["x_cat"])
            loss = bce(out["logits"], batch["y"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
