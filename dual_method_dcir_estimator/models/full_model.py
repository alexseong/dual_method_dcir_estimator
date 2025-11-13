import torch
import torch.nn as nn
from .ocv import AnalyticOCV
from .heads import ParamHead
from .residual import ResidualNN
from .ecm2rc import f_state_2rc

class DCINeuralODE(nn.Module):
    def __init__(self, ocv="analytic", hidden=128, residual_hidden=64, param_scales=None, param_eps=1e-6, dt=1.0):
        super().__init__()
        self.ocv = AnalyticOCV if ocv == "analytic" else ocv
        self.param_head = ParamHead(hidden=hidden, eps=param_eps, scales=param_scales)
        self.residual = ResidualNN(hidden=residual_hidden)
        self.dt = dt
    
    def forward(self, V, I, Tz, soc0=None):
        """
        Rollout over a window.
        Inputs: V(B,H), I(B,H), Tz(B,H)
        soc0: (B,) or None (default 0.8)
        Returns: dict with V_pred, states, params per-step
        """
        device = V.device
        B, H = V.shape

        if soc0 is None:
            soc = torch.full
