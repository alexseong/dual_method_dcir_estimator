import torch
import torch.nn as nn

class ResidualNN(nn.Module):
    """ ΔV residual head: inputs [vc1, vc2, SOC, I, Tz] -> scalar ΔV """    
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, vc1, vc2, SOC, I, Tz):
        x = torch.stack([vc1, vc2, SOC, I, Tz], dim=-1)
        return self.net(x).squeeze(-1)