import torch
import torch.nn as nn
import torch.nn.functional as F

def make_mlp(d_in: int, d_hidden: int, n_layers: int, dropout: float):
    layers = []
    d = d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, d_hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = d_hidden
    return nn.Sequential(*layers), d_hidden

class DragonNetContinuousAdvanced(nn.Module):
    def __init__(
        self,
        n_num: int,
        cat_cardinalities: list[int],
        emb_dim: int,
        d_hidden: int,
        n_shared_layers: int,
        dropout: float,
        min_sigma: float,
        use_t_mlp: bool,
        t_mlp_hidden: int,
        t_mlp_layers: int,
        use_gamma: bool,
        use_cat_offset: bool,
        cat_offset_hidden: int,
    ):
        super().__init__()
        self.min_sigma = float(min_sigma)
        self.use_t_mlp = bool(use_t_mlp)
        self.use_gamma = bool(use_gamma)
        self.use_cat_offset = bool(use_cat_offset)

        self.embs = nn.ModuleList([nn.Embedding(c, emb_dim) for c in cat_cardinalities])
        d_cat = emb_dim * len(cat_cardinalities)
        d_in = n_num + d_cat

        self.shared, d_z = make_mlp(d_in, d_hidden, n_shared_layers, dropout)

        if self.use_t_mlp:
            self.t_mlp, d_t = make_mlp(1, t_mlp_hidden, t_mlp_layers, dropout)
        else:
            self.t_mlp = None
            d_t = 1

        self.q_head = nn.Sequential(
            nn.Linear(d_z + d_t, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

        self.gps_mu = nn.Linear(d_z, 1)
        self.gps_rho = nn.Linear(d_z, 1)

        if self.use_gamma:
            self.gamma_net = nn.Sequential(
                nn.Linear(d_z, d_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_hidden, 1),
            )
        else:
            self.gamma_net = None

        if self.use_cat_offset:
            self.cat_offset_net = nn.Sequential(
                nn.Linear(d_in, cat_offset_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(cat_offset_hidden, 1),
            )
        else:
            self.cat_offset_net = None

    def encode_x(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        return torch.cat([x_num] + embs, dim=1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor, t: torch.Tensor):
        x = self.encode_x(x_num, x_cat)
        z = self.shared(x)

        mu = self.gps_mu(z)
        sigma = F.softplus(self.gps_rho(z)) + self.min_sigma

        if self.use_t_mlp:
            t_feat = self.t_mlp(t)
        else:
            t_feat = t

        logits = self.q_head(torch.cat([z, t_feat], dim=1))

        if self.use_gamma:
            gamma = self.gamma_net(z)
            logits = logits + gamma * t

        if self.use_cat_offset:
            logits = logits + self.cat_offset_net(x)

        return {"x": x, "z": z, "logits": logits, "mu": mu, "sigma": sigma}
