import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.utils import set_seed, get_device
from src.data import generate_synthetic, CausalDataset
from src.model import DragonNetContinuous

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    dcfg = cfg["data"]

    x_num_tr, x_cat_tr, t_tr, y_tr = generate_synthetic(
        n=dcfg["n_train"],
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        seed=cfg["seed"]
    )

    train_ds = CausalDataset(x_num_tr, x_cat_tr, t_tr, y_tr)
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model = DragonNetContinuous(
        n_num=dcfg["n_num"],
        cat_cardinalities=dcfg["cat_cardinalities"],
        emb_dim=cfg["model"]["emb_dim"],
        d_hidden=cfg["model"]["d_hidden"],
        n_shared_layers=cfg["model"]["n_shared_layers"],
        dropout=cfg["model"]["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
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
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f}")

if __name__ == "__main__":
    main()
