import torch
import torch.nn as nn
from .ocv import AnalyticOCV
from .heads import ParamHead
from .residual import ResidualNN
from .rk4 import rk4_step
from .ecm2rc import f_state_2rc

class DCIRNeuralODE(nn.Module):
    def __init__(self, ocv="analytic", hidden=128, residual_hidden=64, param_scales=None, param_eps=1e-6, dt=1.0):
        super().__init__()
        self.ocv = AnalyticOCV() if ocv == "analytic" else ocv
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
            soc = torch.full((B,), 0.8, device=device)
        else:
            soc = soc0

        vc1 = torch.zeros(B, device=device)
        vc2 = torch.zeros(B, device=device)

        V_pred = []
        R0_list, R1_list, C1_list, R2_list, C2_list = [], [], [], [], []
        vc1_list, vc2_list, soc_list = [], [], []

        for k in range(H):
            R0, R1, C1, R2, C2 = self.param_head(soc, Tz[:, k])
            V_ecm = self.ocv(soc) - R0*I[:, k] - vc1 -vc2
            dV = self.residual(vc1, vc2, soc, I[:, k], Tz[:, k])
            Vp = V_ecm + dV
            V_pred.append(Vp)

            # Collect params
            R0_list.append(R0)
            R1_list.append(R1)
            C1_list.append(C1)
            R2_list.append(R2)
            C2_list.append(C2)
            vc1_list.append(vc1)
            vc2_list.append(vc2)
            soc_list.append(soc)

            # RK4 step for states
            x = torch.stack([vc1, vc2, soc], dim=-1)
            x = rk4_step(x, I[:k], self.dt, f_state_2rc, R1, C1, R2, C2)
            vc1, vc2, soc = x[..., 0], x[..., 1], x[..., 2]
            soc = torch.clamp(soc, 0.0, 1.0)

        V_pred = torch.stack(V_pred, dim=1)
        out = dict(
            V_pred=V_pred,
            params=dict(
                R0=torch.stack(R0_list,1),
                R1=torch.stack(R1_list,1),
                C1=torch.stack(C1_list,1),
                R2=torch.stack(R2_list,1),
                C2=torch.stack(C2_list,1)
            ),
            states=dict(
                vc1=torch.stack(vc1_list, 1),
                vc2=torch.stack(vc2_list, 1),
                soc=torch.stack(soc_list, 1)
            )
        )

        return out