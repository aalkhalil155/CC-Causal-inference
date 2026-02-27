import numpy as np
import torch
from torch.utils.data import Dataset

def generate_synthetic(
    n: int,
    n_num: int,
    cat_cardinalities: list[int],
    treatment_noise: float = 0.35,
    outcome_noise: float = 0.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)

    x_num = rng.normal(size=(n, n_num)).astype(np.float32)

    x_cat_cols = []
    for c in cat_cardinalities:
        x_cat_cols.append(rng.integers(0, c, size=(n, 1), dtype=np.int64))
    x_cat = np.concatenate(x_cat_cols, axis=1)

    s1 = (x_num[:, 0] - 0.6 * x_num[:, 1] + 0.35 * x_num[:, 2]).astype(np.float32)
    s2 = (np.tanh(x_num[:, 3]) + 0.6 * np.sin(x_num[:, 4]) - 0.3 * x_num[:, 5]).astype(np.float32)

    cat_effect = np.zeros(n, dtype=np.float32)
    for j, c in enumerate(cat_cardinalities):
        cat_effect += (0.12 / (j + 1)) * (x_cat[:, j] / max(c - 1, 1)).astype(np.float32)

    t = (0.65 * s1 + 0.45 * s2 + cat_effect + rng.normal(scale=treatment_noise, size=n)).astype(np.float32)

    group_boost = (x_cat[:, 0] == 0).astype(np.float32) + (x_cat[:, 1] == 1).astype(np.float32)
    sens = (0.7 + 0.5 * group_boost + 0.25 * (x_num[:, 6] > 0).astype(np.float32)).astype(np.float32)

    base = (-0.15 + 0.8 * s1 + 0.55 * s2 + 0.25 * np.tanh(x_num[:, 7])).astype(np.float32)

    logits = base - sens * t
    if outcome_noise > 0:
        logits += rng.normal(scale=outcome_noise, size=n).astype(np.float32)

    p = 1.0 / (1.0 + np.exp(-logits))
    y = rng.binomial(1, p).astype(np.float32)

    return x_num, x_cat, t, y

class CausalDataset(Dataset):
    def __init__(self, x_num, x_cat, t, y):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.t = torch.tensor(t, dtype=torch.float32).view(-1, 1)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return self.x_num.shape[0]

    def __getitem__(self, idx):
        return {
            "x_num": self.x_num[idx],
            "x_cat": self.x_cat[idx],
            "t": self.t[idx],
            "y": self.y[idx],
        }
