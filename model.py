import torch
import torch.nn as nn
import torch.nn.functional as F

class DragonNetContinuous(nn.Module):
    def __init__(self, n_num, cat_cardinalities, emb_dim, d_hidden, n_shared_layers, dropout, min_sigma=0.05):
        super().__init__()
        self.min_sigma = min_sigma

        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        d_cat = emb_dim * len(cat_cardinalities)
        d_in = n_num + d_cat

        layers = []
        d = d_in
        for _ in range(n_shared_layers):
            layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
            d = d_hidden
        self.shared = nn.Sequential(*layers)

        self.outcome_head = nn.Linear(d_hidden, 1)
        self.gps_mu = nn.Linear(d_hidden, 1)
        self.gps_rho = nn.Linear(d_hidden, 1)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        x = torch.cat([x_num] + embs, dim=1)

        z = self.shared(x)

        logits = self.outcome_head(z)
        mu = self.gps_mu(z)
        sigma = F.softplus(self.gps_rho(z)) + self.min_sigma

        return {"logits": logits, "mu": mu, "sigma": sigma}
