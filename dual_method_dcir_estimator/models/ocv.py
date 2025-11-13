import torch
import torch.nn as nn

class AnalyticOCV(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, soc: torch.Tensor) -> torch.Tensor:
        # Smooth surrogate; replace with lookup if needed
        return 3.0 + 1.2*soc - 0.1*torch.sin(8*torch.pi*soc)

class TableOCV(nn.Module):
    def __init__(self, soc_vec, ocv_vec):
        super().__init__()
        self.register_buffer("soc", soc_vec) # Refer soc_vec tensor, but it is non-learnable state
        self.register_buffer("ocv", ocv_vec) # Refer ocv_vec tensor, but it is non-learnable state

    def forward(self, soc: torch.Tensor) -> torch.Tensor:
        # linear interp
        soc_clamped = torch.clamp(soc, 0.0, 1.0)
        idx = torch.clamp(((soc_clamped - self.soc[0]) / (self.soc[1] - self.soc[0])).long(), 0, len(self.soc)-2)
        s0 = self.soc[idx]; s1 = self.soc[idx+1]
        v0 = self.ocv[idx]; v1 = self.ocv[idx+1]
        w = (soc_clamped - s0) / (s1 - s0 + 1e-12)
        return v0 + w * (v1 - v0)