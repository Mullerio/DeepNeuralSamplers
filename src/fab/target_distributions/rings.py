# Rings
import torch
import torch.nn as nn
from fab.target_distributions.base import TargetDistribution

class PolarTransform(torch.distributions.transforms.Transform):
    """Polar transformation"""

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, PolarTransform)

    def _call(self, x):
        return torch.stack([
            x[..., 0] * torch.cos(x[..., 1]),
            x[..., 0] * torch.sin(x[..., 1])
        ], dim=-1)

    def _inverse(self, y):
        radius = torch.norm(y, p=2, dim=-1)
        radius = torch.clamp(radius, min=1e-6)
        
        x = torch.stack([
            radius,
            torch.atan2(y[..., 1], y[..., 0])
        ], dim=-1)
        x[..., 1] = x[..., 1] + (x[..., 1] < 0).type_as(y) * (2 * torch.pi)
        return x


    ### changed from original source code here!
    def log_abs_det_jacobian(self, x, y):
        #we clamp to avoid log(0)
        radius = torch.clamp(torch.abs(x[..., 0]), min=1e-6)
        return torch.log(radius)


class Rings(nn.Module, TargetDistribution):
    """Rings distribution"""

    def __init__(self, num_modes=4, radius=1.0, sigma=0.15, use_gpu=True, 
                 n_test_set_samples=1000, validate_args=False):
        """Constructor

        The distribution is centered at 0

        Args:
            num_modes (int): Number of circles (default is 4)
            radius (float): Radius of the smallest circle (default is 1.0)
            sigma (float): Width of the circles (default is 0.15)
            use_gpu (bool): Whether to use GPU (default is True)
            n_test_set_samples (int): Number of test samples (for compatibility)
            validate_args (bool): Validation args
        """

        super().__init__()
        self.num_modes = num_modes
        self.radius = radius
        self.sigma = sigma
        self.dim = 2  
        self.n_test_set_samples = n_test_set_samples
        self.device = "cuda" if (use_gpu and torch.cuda.is_available()) else "cpu"
        
        self.dummy_param = torch.nn.Parameter(torch.empty(0, device=self.device))
        
        self.radius_dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(
                torch.ones((num_modes,), device=self.device)),
            component_distribution=torch.distributions.Normal(
                loc=radius * (torch.arange(num_modes).to(self.device) + 1),
                scale=sigma
            )
        )
        # Make the angle distribution

        self.angle_dist = torch.distributions.Uniform(
            low=torch.zeros((1,), device=self.device).squeeze(),
            high=2 * torch.pi * torch.ones((1,), device=self.device).squeeze()
        )
        # Make the polar transform
        self.transform = PolarTransform()
        
        # Set the extreme values
        self.x_min = - radius * num_modes - sigma
        self.x_max = radius * num_modes + sigma
        self.y_min = - radius * num_modes - sigma
        self.y_max = radius * num_modes + sigma
        
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
    
    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples,))

    def sample(self, sample_shape=torch.Size()):
        """Sample the distribution

        Args:
            sample_shape (tuple of int): Shape of the samples

        Returns
            samples (torch.Tensor of shape (*sample_shape, 2)): Samples
        """

        r = self.radius_dist.sample(sample_shape)
        theta = self.angle_dist.sample(sample_shape)
        if len(sample_shape) == 0:
            x = torch.FloatTensor([r, theta])
        else:
            x = torch.stack([r, theta], dim=1)
        return self.transform(x)

    def log_prob2(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """

        x = self.transform.inv(value)
        
        log_p_radius = self.radius_dist.log_prob(x[..., 0])
        log_p_angle = self.angle_dist.log_prob(x[..., 1])
        log_det_jac = self.transform.log_abs_det_jacobian(x, value)
        
        log_prob = log_p_radius + log_p_angle - log_det_jac
        
        log_prob = torch.clamp(log_prob, min=-1e10, max=1e10)
        
        return log_prob
    
    def log_prob(self, value):
        """Evaluate the log-likelihood of the distribution

        Args:
            value (torch.Tensor of shape (batch_size, 2)): Sample

        Returns
            log_prob (torch.Tensor of shape (batch_size,)): Log-likelihood of the samples
        """
        # Compute radius directly in Cartesian coordinates
        radius = torch.sqrt(value[..., 0]**2 + value[..., 1]**2)
        radius = torch.clamp(radius, min=1e-6)  #clamp here for the rest 
        
        
        log_p_radius = self.radius_dist.log_prob(radius)
        
        #this is known since angle is uniform distribution
        log_p_angle = -torch.log(torch.tensor(2 * torch.pi, device=value.device, dtype=value.dtype))
        
        log_det_jac = torch.log(radius)
        
        log_prob = log_p_radius + log_p_angle - log_det_jac
        
        log_prob = torch.clamp(log_prob, min=-1e10, max=1e10)
        
        return log_prob

    def _apply(self, fn):
        """Apply the fn function on the distribution

        Args:
            fn (function): Function to apply on tensors
        """

        new_self = super(Rings, self)._apply(fn)
        # Radius distribution
        new_self.radius_dist.mixture_distribution.probs = fn(
            new_self.radius_dist.mixture_distribution.probs)
        new_self.radius_dist.component_distribution.loc = fn(
            new_self.radius_dist.component_distribution.loc)
        new_self.radius_dist.component_distribution.scale = fn(
            new_self.radius_dist.component_distribution.scale)
        # Angle distribution
        new_self.angle_dist.low = fn(new_self.angle_dist.low)
        new_self.angle_dist.high = fn(new_self.angle_dist.high)
        return new_self

        