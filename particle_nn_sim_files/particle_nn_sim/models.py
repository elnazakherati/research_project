import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim=12, hidden=128, out_dim=8, dropout=0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers += [
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        # mild stabilization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class InteractionNet2(nn.Module):
    """Tiny interaction network for N=2 (pair interaction only)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.phi_node = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        edge_in = 2*hidden + 5  # h_i,h_j, dx,dy,dvx,dvy,dist
        self.phi_edge = nn.Sequential(
            nn.Linear(edge_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.phi_update = nn.Sequential(
            nn.Linear(2*hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.phi_out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, X):
        x0 = X[:, 0:6]
        x1 = X[:, 6:12]

        h0 = self.phi_node(x0)
        h1 = self.phi_node(x1)

        p0 = X[:, 0:2]; v0 = X[:, 2:4]
        p1 = X[:, 6:8]; v1 = X[:, 8:10]

        dp01 = p1 - p0
        dv01 = v1 - v0
        dist = torch.sqrt((dp01**2).sum(dim=1, keepdim=True) + 1e-12)

        dp10 = -dp01
        dv10 = -dv01

        edge10 = torch.cat([h0, h1, dp10, dv10, dist], dim=1)
        edge01 = torch.cat([h1, h0, dp01, dv01, dist], dim=1)

        m10 = self.phi_edge(edge10)
        m01 = self.phi_edge(edge01)

        h0p = self.phi_update(torch.cat([h0, m10], dim=1))
        h1p = self.phi_update(torch.cat([h1, m01], dim=1))

        y0 = self.phi_out(h0p)
        y1 = self.phi_out(h1p)

        return torch.cat([y0, y1], dim=1)

class ResMLP(nn.Module):
    """
    Residual MLP for dynamics learning.
    Learns corrections on top of free-flight / residual targets.
    """
    def __init__(self, in_dim=12, hidden=256, out_dim=8, blocks=3, dropout=0.0):
        super().__init__()

        self.in_proj = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
            )
            for _ in range(blocks)
        ])

        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, out_dim)

        # Good initialization for residual nets
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h = self.in_proj(x)
        for blk in self.blocks:
            h = h + blk(h)   # residual connection
        h = self.out_norm(h)
        return self.out(h)
