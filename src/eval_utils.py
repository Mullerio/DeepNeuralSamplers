import torch
from torch import nn
from torchdiffeq import odeint
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class cnf_sample(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, states):
        x = states[0]
        logp_z = states[1]
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            dz_dt = self.model(t,x)
            
            dlogp_z_dt = trace_df_dz(dz_dt, x).view(len(x), 1)
           
        return dz_dt, dlogp_z_dt

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()

