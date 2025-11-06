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
        
