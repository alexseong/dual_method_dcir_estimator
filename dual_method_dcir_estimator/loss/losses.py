import torch 
import torch.nn as nn

class VoltageLoss(nn.Module):
    def __init__(self, lambda_sm=5e-4, lambda_res=1e-3):
        super().__init__()
        self.lambda_sm = lambda_sm
        self.lambda_res = lambda_res
        self.mse = nn.MSELoss()

    def forward(self, V_pred, V_true, params, states, residual_mag=None, weight_decay=0.0, Model=None):
        loss_v = self.mse(V_pred, V_true)

        # smoothness on parameters
        def diff_pen(x):
            return((x[:,1:] - x[:,:-1])**2).mean()

        sm = 0.0
        for k in 