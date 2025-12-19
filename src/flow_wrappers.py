"""
Flow model wrappers for different model types.

This module provides consistent wrapper classes for:
- Linear flow models (velocity field only)
- Learned flow models (potential + velocity field)
- Gradient flow models (potential + schedule + beta)

These wrappers are used for both trajectory generation and action computation.
"""

import torch
import torch.nn as nn


class LinearFlowWrapper(nn.Module):
    """Wrapper for linear flow model with velocity field only."""
    
    def __init__(self, velo):
        """
        Args:
            velo: Velocity field network v_θ(x, t)
        """
        super().__init__()
        self.velo = velo
    
    def forward(self, t, x):
        """
        Compute velocity at time t and position x.
        
        Args:
            t: Time (scalar or tensor)
            x: Position tensor [batch_size, dim]
            
        Returns:
            Velocity tensor [batch_size, dim]
        """
        time = t.repeat(x.shape[0])[:, None] if t.dim() == 0 else t
        return self.velo(torch.cat([x, time], 1))


class LearnedFlowWrapper(nn.Module):
    """Wrapper for learned flow with potential and velocity field."""
    
    def __init__(self, psi, velo, target):
        """
        Args:
            psi: Potential network ψ_θ(x, t)
            velo: Velocity field network v_θ(x, t)
            target: Target distribution with log_prob method
        """
        super().__init__()
        self.psi = psi
        self.velo = velo
        self.target = target
    
    def forward(self, t, x):
        """
        Compute velocity as: v = ∇f + v_θ
        where f = t·ψ(x,t) + (1-t)·log p(x)
        
        Args:
            t: Time (scalar or tensor)
            x: Position tensor [batch_size, dim]
            
        Returns:
            Velocity tensor [batch_size, dim]
        """
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            time = t.repeat(x.shape[0])[:, None] if t.dim() == 0 else t
            
            # Compute potential function
            f = time * self.psi(torch.cat([x, time], 1)).reshape(x.shape[0], 1) + \
                (1 - time) * self.target.log_prob(x).reshape(x.shape[0], 1)
            
            # Gradient of potential
            grad = torch.autograd.grad(torch.sum(f), x)[0]
        
        # Learned velocity correction
        v_theta = self.velo(torch.cat([x, time], 1))
        
        return grad + v_theta


class GradientFlowWrapper(nn.Module):
    """Wrapper for gradient flow with action network and schedule."""
    
    def __init__(self, psi, schedule, target, beta_min=0.1, beta_max=20.0):
        """
        Args:
            psi: Action network ψ_θ(x, t)
            schedule: Schedule network for time-dependent potential
            target: Target distribution with log_prob method
            beta_min: Minimum beta value (default: 0.1)
            beta_max: Maximum beta value (default: 20.0)
        """
        super().__init__()
        self.psi = psi
        self.schedule = schedule
        self.target = target
        self.beta_min = beta_min
        self.beta_max = beta_max
    
    def beta(self, t):
        """Compute beta(t) = 0.5 * (beta_min + (beta_max - beta_min) * t)"""
        return 0.5 * (self.beta_min + (self.beta_max - self.beta_min) * t)
    
    def forward(self, t, x):
        """
        Compute velocity as: v = -β(t) * (∇f + x)
        where f = t·ψ(x,t) + [t·s(t) + (1-t)]·log p(x)
        
        Args:
            t: Time (scalar or tensor)
            x: Position tensor [batch_size, dim]
            
        Returns:
            Velocity tensor [batch_size, dim]
        """
        with torch.set_grad_enabled(True):
            x.requires_grad_(True)
            time = t.repeat(x.shape[0])[:, None] if t.dim() == 0 else t
            
            # Compute learned potential
            learned = time * self.psi(torch.cat([x, time], 1)).reshape(x.shape[0], 1) + \
                      ((time) * self.schedule(time).reshape(x.shape[0], 1) + (1 - time)) * \
                      self.target.log_prob(x).reshape(x.shape[0], 1)
            
            # Gradient of learned potential
            grad = torch.autograd.grad(torch.sum(learned), x)[0]
        
        # Apply gradient flow dynamics
        beta_t = self.beta(time)
        return -beta_t * (grad + x)


def create_flow_wrapper(models, model_type, target, beta_min=0.1, beta_max=20.0):
    """
    Factory function to create appropriate flow wrapper.
    
    Args:
        models: Dictionary containing model networks
        model_type: One of "linear", "learned", or "gf"
        target: Target distribution
        beta_min: Minimum beta for gradient flow (default: 0.1)
        beta_max: Maximum beta for gradient flow (default: 20.0)
        
    Returns:
        Flow wrapper instance
    """
    if model_type == "linear":
        return LinearFlowWrapper(models["velo"])
    
    elif model_type == "learned":
        return LearnedFlowWrapper(models["psi"], models["velo"], target)
    
    elif model_type == "gf":
        return GradientFlowWrapper(
            models["psi"], 
            models["schedule"], 
            target,
            beta_min=beta_min,
            beta_max=beta_max
        )
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'linear', 'learned', or 'gf'")
