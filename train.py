import importlib
import inspect
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
            raise TypeError(
                f"Model forward requires '{name}' but it is missing from the batch."
            )

    return model(**forward_kwargs)


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

    model = build_model(dcfg, cfg["model"], device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    bce = torch.nn.BCEWithLogitsLoss()

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model_forward(model, batch)
            loss = bce(out["logits"], batch["y"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"].get("grad_clip", 5.0))
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1} | Loss: {np.mean(losses):.4f}")


if __name__ == "__main__":
    main()
