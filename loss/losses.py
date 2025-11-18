import torch 
import torch.nn as nn

class VoltageLoss(nn.Module):
    def __init__(self, lambda_sm=5e-4, lambda_res=1e-3):
        super().__init__()
        self.lambda_sm = lambda_sm
        self.lambda_res = lambda_res
        self.mse = nn.MSELoss()

    def forward(self, V_pred, V_true, params, states, residual_mag=None, weight_decay=0.0, model=None):
        loss_v = self.mse(V_pred, V_true)

        # smoothness on parameters
        def diff_pen(x):
            return((x[:,1:] - x[:,:-1])**2).mean()

        sm = 0.0
        for k in ["R0", "R1", "C1", "R2", "C2"]:
            sm = sm + diff_pen(params[k])

        # residual penalty (optional): if residual magnitude passed
        res = 0.0
        if residual_mag is not None:
            res = (residual_mag**2).mean()

        # weight decay on model params (manual, optional)
        wd = 0.0
        if weight_decay > 0.0 and model is not None:
            wd = sum( (p**2).sum() for n, p in model.named_parameters() if p.requires_grad) * weight_decay

        return loss_v + self.lambda_sm*sm + self.lambda_res*res + wd, \
            dict(v=loss_v.item(), sm=sm.item(), res=res if isinstance(res, float) else float(res), wd=float(wd))