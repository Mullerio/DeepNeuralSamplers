import torch
from torch import nn
from torchdiffeq import odeint
from typing import Callable, Optional
from fab.utils.utils import *
import math
import numpy as np
from scipy.special import gamma
from torch.special import gammaln

class GeneralizedGaussian:
    """
    For p = 2 exact gaussian, not sure if p != 2 case is as in Mate et al.
    """

    def __init__(self, sigma, p, dim, device='cpu'):
        self.sigma = float(sigma)
        self.p = float(p)
        self.dim = int(dim)
        self.device = device

        if self.p == 2:
            cov = (self.sigma ** 2) * torch.eye(self.dim, device=device)
            self.gaussian = torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.dim, device=device),
                covariance_matrix=cov
            )
            
    def log_prob(self, x):
        if self.p == 2:
            return self.gaussian.log_prob(x)

        r = torch.norm(x, p=self.p, dim=-1)
        d = self.dim

        log_V_dp = (
            d * (torch.log(torch.tensor(2.0, device=self.device)) + gammaln(torch.tensor(1 + 1 / self.p, device=self.device)))
            - gammaln(torch.tensor(1 + d / self.p, device=self.device))
        )

        # log Z = log(V_d,p) + d log sigma + log Γ(d/p) - log p
        logZ = (
            log_V_dp
            + d * torch.log(torch.tensor(self.sigma, device=self.device))
            + gammaln(torch.tensor(d / self.p, device=self.device))
            - torch.log(torch.tensor(self.p, device=self.device))
        )

        return -((r / self.sigma) ** self.p) - logZ


    def sample(self, batch_size, device='cpu'):
        if self.p == 2:
            return self.gaussian.sample((batch_size,)).to(device)

        shape_param = self.dim / self.p
        scale_param = 2.0  

        z = torch.tensor(
            np.random.gamma(shape_param, scale_param, batch_size),
            dtype=torch.float32,
            device=device
        )

        u = torch.randn(batch_size, self.dim, device=device)
        u = u / torch.norm(u, dim=1, keepdim=True)

        return self.sigma * (z.unsqueeze(1) ** (1.0 / self.p)) * u



def beta(t, beta_min, beta_max):
    return 0.5 * (beta_min + (beta_max - beta_min) * (t))

def beta_int(t, beta_min, beta_max):
    return beta_min * t + (beta_max - beta_min) / 2 * t**2


class TorchWrapperGF(nn.Module):
    def __init__(self, psi, schedule, target, beta):
        super().__init__()
        self.psi = psi
        self.schedule = schedule
        self.target = target
        self.beta = beta

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # Handle both scalar and tensor time inputs
            if isinstance(t, (int, float)):
                t = torch.tensor(t, device=x.device, dtype=x.dtype)
            if t.dim() == 0:
                t = t.unsqueeze(0)
            time = t.expand(x.shape[0], 1)
            
            learned = time * self.psi(torch.cat([x, time], 1)).reshape(x.shape[0], 1) + \
                      ((time) * self.schedule(time).reshape(x.shape[0], 1) + (1 - time)) * (self.target.log_prob(x)).reshape(x.shape[0], 1)
            grad = torch.autograd.grad(torch.sum(learned), x, create_graph=True)[0]

        return -self.beta(time) * (grad.reshape(x.shape[0], x.shape[1]) + x.reshape(x.shape[0], x.shape[1]))
    
class TorchWrapper(nn.Module):
    """Wraps velocity model for ODE integration.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            # Handle 0D scalar from odeint, otherwise assume [batch, 1]
            if t.dim() == 0:
                time = t.unsqueeze(0).expand(x.shape[0], 1)
            else:
                time = t
            v = self.model(torch.cat([x, time], 1))
        return v


def step_gf(models, target, args, device, x1_sampler=None):
    """
    GF training step: models has psi, schedule, and optionally Ct
    Returns: loss_tensor (requires_grad=True), metrics dict
    """
    psi = models.psi
    schedule = models.schedule
    Ct = models.Ct
    batch_size = args.batch_size
    dim = args.dim
    x1_sampler = x1_sampler or (lambda b, d, dev: torch.rand((b, d), device=dev) * 100 - 50)
    total_loss = 0.0

    for j in range(args.inner_steps - 1):

        time = torch.rand((batch_size, 1), device=device) * 0.999 + 0.001

        xt = torch.sqrt((1 - torch.exp(-beta_int(time, args.beta_min, args.beta_max)))) * \
             torch.randn((batch_size, dim), device=device) * args.sigma + \
             torch.exp(0.5 * (-beta_int(time, args.beta_min, args.beta_max))) * \
             x1_sampler(batch_size, dim, device)

        xt.requires_grad_(True)
        time.requires_grad_(True)

        Zt = Ct(time).reshape(batch_size, 1) if Ct is not None else 0

        # Clamp log_prob to prevent instability in low-density regions
        log_prob_xt = target.log_prob(xt).reshape(batch_size, 1)
        #log_prob_xt = torch.clamp(log_prob_xt, min=-1e6)
        
        f = (time) * psi(torch.cat([xt, time], 1)).reshape(batch_size, 1) + \
            ((time) * schedule(time).reshape(batch_size, 1) + (1 - time)) * log_prob_xt

        dfdt = torch.autograd.grad(torch.sum(f), time, create_graph=True)[0].reshape(batch_size, 1)
        dfdx = torch.autograd.grad(torch.sum(f), xt, create_graph=True)[0].reshape(batch_size, dim)

        # Compute Laplacian of f
        lap_f = 0.0
        for i in range(dim):
            lap_f = lap_f + torch.autograd.grad((dfdx[:, i]).sum(),
                                                xt, create_graph=True)[0][:, i]

        lap_f = lap_f.reshape(batch_size, 1)

        dot = (dfdx * -1 * (beta(time, args.beta_min, args.beta_max) * (xt + dfdx))).sum(1, keepdims=True)

        residual = (dfdt + Zt + dot.reshape(batch_size, 1) -
                    beta(time, args.beta_min, args.beta_max) * (lap_f + dim))
        

        loss = (residual ** 2).mean()
        total_loss = total_loss + loss

    metrics = {"last_loss": total_loss.detach().item()}
    return total_loss, metrics



def step_learned(models, target, args, device, x1_sampler=None):
    """
    Learned (non-linear) training step: models is TrainingModels with psi, velo, Ct
    Calls backward() inside loop for efficiency (matches original funnel_learned.py)
    Uses GeneralizedGaussian for initial distribution.
    
    If use_x1_sampler=True, samples x_t from x1_sampler instead of integrating trajectories.
    In this case, inner_steps is automatically set to 2.
    """
    psi = models.psi
    velo = models.velo
    Ct = models.Ct
    batch_size = args.batch_size
    dim = args.dim
    p = getattr(args, 'p', 2.0)  # Default to p=2.0 (standard Gaussian) if not specified
    use_x1_sampler = getattr(args, 'use_x1_sampler', False)
    
    # Initialize generalized Gaussian
    gen_gauss = GeneralizedGaussian(args.sigma, p, dim, device=device)
    x1_sampler = x1_sampler or (lambda b, d, dev: torch.rand((b, d), device=dev) * 100 - 50)

    xt = gen_gauss.sample(batch_size, device=device)
    
    if use_x1_sampler:
        # Sample mode: sample time uniformly like in GF, single step per batch
        num_steps = 2
    else:
        # Trajectory mode: use configured inner_steps
        num_steps = args.inner_steps

    total_loss = 0.0
    for step in range(num_steps - 1):
        if use_x1_sampler:
            # Sample random time for each step (like GF)
            time = torch.rand((batch_size, 1), device=device) * 0.999 + 0.001
            # Sample x directly from x1_sampler (evaluate over target domain)
            xt = x1_sampler(batch_size, dim, device)
        else:
            # Trajectory mode: use discretized time steps
            times = torch.linspace(0, 1, args.inner_steps, device=device)
            t_scalar = times[step].item()
            t1_scalar = times[step + 1].item()
            time = torch.full((batch_size, 1), t1_scalar, dtype=torch.float32, device=device)
            
            # Integrate using ODE solver
            with torch.no_grad():
                traj = odeint(TorchWrapper(velo), xt, torch.tensor([t_scalar, t1_scalar]).type(torch.float32).to(device),
                              atol=args.ode_atol, rtol=args.ode_rtol, method='dopri5')
                xt = traj[-1].detach()

        xt.requires_grad_(True)
        time.requires_grad_(True)
        #compute the normalizing constant depending on if we train C_t or not 
        Zt = Ct(time).reshape(batch_size, 1) if Ct is not None else 0

        learned = (time) * (1 - time) * psi(torch.cat([xt, time], 1)).reshape(batch_size, 1) 
        f_0 = gen_gauss.log_prob(xt).reshape(xt.shape[0], 1)
        f_1 = target.log_prob(xt).reshape(xt.shape[0], 1)
        f = learned + (time) * f_1 + (1 - time) * f_0

        dfdt = (torch.autograd.grad(torch.sum(learned), time, create_graph=True)[0].reshape(batch_size, 1)) + f_1 - f_0
        dfdx = torch.autograd.grad(torch.sum(f), xt, create_graph=True)[0].reshape(batch_size, dim)
        vt = velo(torch.cat([xt, time], 1)).reshape(batch_size, dim)

        div = 0.0
        for i in range(xt.shape[1]):  
            d2vdx2_i, = torch.autograd.grad((vt[:, i]).sum(), xt, create_graph=True)
            div += d2vdx2_i[:, i]
        
        dot = (dfdx * vt).sum(1, keepdims=True)
        residual = dfdt.reshape(batch_size, 1) + Zt + dot.reshape(batch_size, 1) + div.reshape(batch_size, 1)
        loss = (torch.abs(residual) + (residual) ** 2).mean()
        loss = loss / (args.inner_steps - 1)
        total_loss += loss.item()
        
        loss.backward()

    metrics = {"last_loss": total_loss}
    return loss, metrics  # gradients already computed via backward() in loop

def step_linear(models, target, args, device, x1_sampler=None):
    """
    Linear training step: models has velo and Ct
    Calls backward() inside loop for efficiency (matches original funnel_linear.py)
    Uses GeneralizedGaussian for initial distribution.
    
    If use_x1_sampler=True, samples x_t from x1_sampler instead of integrating trajectories.
    In this case, inner_steps is automatically set to 2.
    """
    velo = models.velo
    Ct = models.Ct
    batch_size = args.batch_size
    dim = args.dim
    p = getattr(args, 'p', 2.0)  # Default to p=2.0 (standard Gaussian) if not specified
    use_x1_sampler = getattr(args, 'use_x1_sampler', False)
    
    # Initialize generalized Gaussian
    gen_gauss = GeneralizedGaussian(args.sigma, p, dim, device=device)
    x1_sampler = x1_sampler or (lambda b, d, dev: torch.rand((b, d), device=dev) * 100 - 50)

    xt = gen_gauss.sample(batch_size, device=device)
    
    if use_x1_sampler:
        # Sample mode: sample time uniformly like in GF, single step per batch
        num_steps = 1
    else:
        # Trajectory mode: use configured inner_steps
        num_steps = args.inner_steps - 1
    
    total_loss = 0.0
    for step in range(num_steps):
        if use_x1_sampler:
            # Sample random time for each step (like GF)
            time = torch.rand((batch_size, 1), device=device) * 0.999 + 0.001
            # Sample x directly from x1_sampler (evaluate over target domain)
            xt = x1_sampler(batch_size, dim, device)
        else:
            # Trajectory mode: use discretized time steps
            times = torch.linspace(0, 1, args.inner_steps, device=device)
            t_scalar = times[step].item()
            t1_scalar = times[step + 1].item()
            time = torch.full((batch_size, 1), t1_scalar, dtype=torch.float32, device=device)
            
            # Integrate using ODE solver
            with torch.no_grad():
                traj = odeint(TorchWrapper(velo), xt, torch.tensor([t_scalar, t1_scalar]).type(torch.float32).to(device),
                              atol=args.ode_atol, rtol=args.ode_rtol, method='dopri5')
                xt = traj[-1].detach()

        xt.requires_grad_(True)
        time.requires_grad_(True)
        
        Zt = Ct(time).reshape(batch_size, 1) if Ct is not None else 0

        f_0 = gen_gauss.log_prob(xt).reshape(xt.shape[0], 1)
        f_1 = target.log_prob(xt).reshape(xt.shape[0], 1)
        f = (time) * f_1 + (1 - time) * f_0

        dfdt = f_1 - f_0
        dfdx = torch.autograd.grad(torch.sum(f), xt, create_graph=True)[0].reshape(batch_size, dim)
        vt = velo(torch.cat([xt, time], 1)).reshape(batch_size, dim)

        div = 0.0
        for i in range(xt.shape[1]):  
            d2vdx2_i, = torch.autograd.grad((vt[:, i]).sum(), xt, create_graph=True)
            div += d2vdx2_i[:, i]
        
        dot = (dfdx * vt).sum(1, keepdims=True)
        residual = dfdt.reshape(batch_size, 1) + Zt + dot.reshape(batch_size, 1) + div.reshape(batch_size, 1)
        loss = (torch.abs(residual) + (residual) ** 2).mean()
        loss = loss / (args.inner_steps - 1)
        total_loss += loss.item()
        
        loss.backward()

    metrics = {"last_loss": total_loss}
    return loss, metrics  # gradients already computed via backward() in loop