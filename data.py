import numpy as np
import torch
from torch.utils.data import Dataset

def generate_synthetic(n, n_num, cat_cardinalities, treatment_noise=0.25, outcome_noise=0.0, seed=0):
    rng = np.random.default_rng(seed)

    x_num = rng.normal(size=(n, n_num)).astype(np.float32)

    x_cat = []
    for c in cat_cardinalities:
        x_cat.append(rng.integers(0, c, size=(n, 1), dtype=np.int64))
    x_cat = np.concatenate(x_cat, axis=1)

    s1 = x_num[:, 0] - 0.5 * x_num[:, 1]
    s2 = np.tanh(x_num[:, 2])

    cat_effect = 0.1 * (x_cat[:, 0] / max(cat_cardinalities[0] - 1, 1)).astype(np.float32)

    t = (0.6 * s1 + 0.4 * s2 + cat_effect + rng.normal(scale=treatment_noise, size=n)).astype(np.float32)

    sens = 0.8 + 0.6 * (x_cat[:, 0] == 0).astype(np.float32)
    base = -0.2 + 0.7 * s1 + 0.5 * s2

    logits = base - sens * t
    if outcome_noise > 0:
        logits += rng.normal(scale=outcome_noise, size=n).astype(np.float32)

    p = 1 / (1 + np.exp(-logits))
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
