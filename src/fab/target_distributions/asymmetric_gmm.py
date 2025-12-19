"""Asymmetric Gaussian Mixture Model target distribution."""
from typing import Optional, Dict
import torch
import torch.nn as nn
from fab.target_distributions.base import TargetDistribution


class AsymmetricGMM(nn.Module, TargetDistribution):
    """
    Asymmetric Gaussian Mixture: 1/3 N([4,4], I) + 2/3 N([-m,-m], I)
    
    Two-component Gaussian mixture with different weights and means.
    First component: weight=1/3, mean=[4, 4, 0, ...]
    Second component: weight=2/3, mean=[-m, -m, 0, ...]
    Both components have identity covariance matrices.
    
    Args:
        dim: Dimension of the distribution
        mean_offset: Absolute value of the mean for the second component (default: 8.0)
                     Second component will have mean [-mean_offset, -mean_offset, 0, ...]
        use_gpu: Whether to use GPU (default: True)
        n_test_set_samples: Number of test samples (for compatibility with other targets)
    """
    def __init__(self, dim=2, mean_offset=8.0, use_gpu=True, 
                 n_test_set_samples=1000, **kwargs):
        super().__init__()
        self.dim = dim
        self.mean_offset = float(mean_offset)
        self.n_test_set_samples = n_test_set_samples
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        
        # Component weights
        self.register_buffer("weights", torch.tensor([1/3, 2/3]))
        
        # Component means: [4, 4, 0, ...] and [-m, -m, 0, ...]
        means = torch.zeros(2, dim)
        means[0, 0] = 4.0
        means[0, 1] = 4.0
        means[1, 0] = -self.mean_offset
        means[1, 1] = -self.mean_offset
        self.register_buffer("means", means)
        
        # Component covariances (identity matrices)
        self.register_buffer("cov", torch.eye(dim))
        self.register_buffer("cov_inv", torch.eye(dim))
        self.register_buffer("log_det_cov", torch.tensor(0.0))  # log det of identity = 0
        
        # Normalization constant for multivariate Gaussian
        import math
        self.register_buffer("log_norm", 
                           torch.tensor(-0.5 * dim * math.log(2 * math.pi)))
        
        self.to(self.device)
        
    def to(self, device):
        """Move to device"""
        if isinstance(device, str):
            self.device = device
            if device == "cuda":
                if torch.cuda.is_available():
                    super().cuda()
            else:
                super().cpu()
        else:
            self.device = str(device)
            super().to(device)
        return self
        
    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of the mixture.
        
        Args:
            x: Input samples of shape [batch_size, dim]
            
        Returns:
            log_prob: Log probabilities of shape [batch_size]
        """
        batch_size = x.shape[0]
        
        # Compute log probability for each component
        log_probs = []
        for i in range(2):
            # x - mu_i
            diff = x - self.means[i].unsqueeze(0)  # [batch_size, dim]
            
            # -0.5 * (x - mu)^T * Sigma^{-1} * (x - mu)
            mahalanobis = -0.5 * torch.sum(diff * torch.matmul(diff, self.cov_inv), dim=1)
            
            # Add normalization and weight
            log_prob_i = torch.log(self.weights[i]) + self.log_norm + self.log_det_cov + mahalanobis
            log_probs.append(log_prob_i)
        
        # Log-sum-exp for mixture
        log_probs = torch.stack(log_probs, dim=1)  # [batch_size, 2]
        return torch.logsumexp(log_probs, dim=1)
    
    def sample(self, shape=(1,)):
        """
        Sample from the mixture.
        
        Args:
            shape: Shape of samples to generate (batch_size,) or batch_size
            
        Returns:
            samples: Samples of shape [batch_size, dim]
        """
        batch_size = shape[0] if isinstance(shape, tuple) else shape
        
        # Sample component assignments
        component_probs = self.weights.cpu()
        components = torch.multinomial(component_probs, batch_size, replacement=True)
        
        # Sample from each component
        samples = torch.zeros(batch_size, self.dim, device=self.device)
        for i in range(2):
            mask = (components == i)
            n_samples = mask.sum().item()
            if n_samples > 0:
                # Sample from N(mu_i, I)
                component_samples = torch.randn(n_samples, self.dim, device=self.device) + self.means[i]
                samples[mask] = component_samples
        
        return samples
    
    @property
    def test_set(self) -> torch.Tensor:
        """Generate test set samples"""
        return self.sample((self.n_test_set_samples,))
    
    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                          log_q_fn=None,
                          batch_size: Optional[int] = None) -> Dict:
        """
        Compute performance metrics (placeholder for compatibility).
        
        Args:
            samples: Samples from the model
            log_w: Log importance weights
            log_q_fn: Log probability function (optional)
            batch_size: Batch size for processing (optional)
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        if log_q_fn is not None:
            # Compute forward KL if we have the model's log prob
            log_p = self.log_prob(samples)
            log_q = log_q_fn(samples)
            kl_forward = torch.mean(log_p - log_q)
            metrics["kl_forward"] = kl_forward.detach().cpu().item()
            
        return metrics
