import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP2(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
        )

    def forward(self, x):
        return self.net(x)

class MLP3(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
           
        )

    def forward(self, x):
        return self.net(x)

class MLP4(torch.nn.Module):
    def __init__(self, dim, out_dim=None, w=256, time_varying=False):
        super().__init__()
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim + (1 if time_varying else 0), w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, w),
            torch.nn.SiLU(),
            torch.nn.Linear(w, out_dim),
           
        )

    def forward(self, x):
        return self.net(x)


class TrainingModels(nn.Module):
    def __init__(self, psi=None, velo=None, Ct=None, schedule=None):
        super().__init__()
        # register modules if provided
        self.psi = psi
        self.velo = velo
        self.Ct = Ct
        self.schedule = schedule