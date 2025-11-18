import torch
import torch.nn as nn

class ParamHead(nn.Module):
    """
    Maps [SOC, Tz] -> positive parameters (R0,R1,C1,R2,C2).
    Uses Softplus + scaling to realistic magnitudes.
    """    
    def __init__(self, hidden=128, eps=1e-6, scales=None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 5)
        )
        self.softplus = nn.Softplus()
        self.eps = eps
        
        if scales is None:
            scales = dict(r0=5e-3, r1=5e-3, r2=5e-3, c1=5e-3, c2=5e-3)
        self.scales = scales

    def forward(self, soc, tz):
        z = torch.stack([soc, tz], dim=-1)
        raw = self.mlp(z)
        raw = self.softplus(raw) + self.eps

        R0 = raw[..., 0] * self.scales["r0"]
        R1 = raw[..., 1] * self.scales["r1"]
        C1 = raw[..., 2] * self.scales["c1"]
        R2 = raw[..., 3] * self.scales["r2"]
        C2 = raw[..., 4] * self.scales["c2"]

        return R0, R1, C1, R2, C2