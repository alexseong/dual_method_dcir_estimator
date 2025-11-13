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
            nn.SiLU(hidden, hidden),
            nn.Linear(2, hidden),
            nn.Linear(hidden, 5)
        )
        self.softplus = nn.Softplus()
        self.eps = eps
        
        if scales is None:
            scales = dict(r0=5e-3, r1=5e-3, r2=5e-3, c1=5e-3, c2=5e-3)
        self.scales = scales

    # def forward(self, soc: torch.Tensor) -> torch.Tensor:
    #     # Smooth surrogate; replace with lookup if needed
    #     return 3.0 + 1.2*soc - 0.1*torch.sin(8*torch.pi*soc)