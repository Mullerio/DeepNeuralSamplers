from typing import Optional, Dict
from fab.types_ import LogProbFunc
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from fab.target_distributions.base import TargetDistribution
from fab.utils.numerical import MC_estimate_true_expectation, quadratic_function, \
    importance_weighted_expectation, effective_sample_size_over_p, setup_quadratic_function

class Funnel(nn.Module, TargetDistribution):
    """
    Neal's Funnel distribution:
      y ~ Normal(0, sigma^2)
      x_i | y ~ Normal(0, exp(y) / divisor)   for i=1..dim-1
      
    If divide_variance_by_two=True: variance = exp(y)/2 
    If divide_variance_by_two=False: variance = exp(y) 
    """

    def __init__(self, dim: int, sigma: float = 1.0,
                 n_test_set_samples: int = 1000,
                 true_expectation_estimation_n_samples: int = int(1e6),
                 use_gpu: bool = True,
                 divide_variance_by_two: bool = True):
        super(Funnel, self).__init__()
        assert dim >= 2, "Funnel requires dim >= 2"
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.to(self.device)
        self.dim = dim
        self.register_buffer("sigma", torch.tensor(sigma))
        self.n_test_set_samples = n_test_set_samples
        self.divide_variance_by_two = divide_variance_by_two

        # Precompute constants
        self.register_buffer("_log_2pi", torch.tensor(math.log(2 * math.pi)))
        self.register_buffer("_sigma2", self.sigma ** 2)
        self.register_buffer("_var_divisor", torch.tensor(2.0 if divide_variance_by_two else 1.0))

        # Define expectation function (for diagnostics)
        self.expectation_function = quadratic_function
        self.register_buffer(
            "true_expectation",
            MC_estimate_true_expectation(self,
                                         self.expectation_function,
                                         true_expectation_estimation_n_samples)
        )

        

    def to(self, device):
        if device == "cuda":
            if torch.cuda.is_available():
                self.cuda()
        else:
            self.cpu()

    @property
    def test_set(self) -> torch.Tensor:
        return self.sample((self.n_test_set_samples,))

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x) for Funnel distribution.
        x: (..., dim)   
        returns: (...,)
        """ 
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected last dim {self.dim}, got {x.shape[-1]}")

        y = x[..., 0]         # (...,)
        xs = x[..., 1:]       # (..., dim-1)

        # log p(y) = -0.5 * (log(2πσ²) + y²/σ²)
        log_py = -0.5 * (self._log_2pi + torch.log(self._sigma2) + (y ** 2) / self._sigma2)

        # log p(x | y) depends on variance divisor
        # variance = exp(y) / divisor, so log_var = y - log(divisor)
        # precision = divisor * exp(-y)
        xs_sq = (xs ** 2).sum(dim=-1)             # (...,)
        log_var = y - torch.log(self._var_divisor)
        precision = self._var_divisor * torch.exp(-y)
        log_px_cond = -0.5 * ((self.dim - 1) * (self._log_2pi + log_var) + precision * xs_sq)

        return log_py + log_px_cond

    def sample(self, shape=(1,)) -> torch.Tensor:
        """
        Draw samples from the funnel distribution.
        shape: leading shape, returns tensor of shape (*shape, dim)
        """
        if isinstance(shape, int):
            shape = (shape,)

        # sample y ~ N(0, sigma^2)
        y = torch.randn(*shape, device=self.device) * self.sigma  # (...,)

        # sample xs | y ~ N(0, exp(y)/divisor)
        std_x = torch.sqrt(torch.exp(y) / self._var_divisor).unsqueeze(-1)     # (..., 1)
        xs = torch.randn(*shape, self.dim - 1, device=self.device) * std_x

        return torch.cat([y.unsqueeze(-1), xs], dim=-1)

    def evaluate_expectation(self, samples: torch.Tensor, log_w: torch.Tensor):
        expectation = importance_weighted_expectation(self.expectation_function,
                                                      samples, log_w)
        true_expectation = self.true_expectation.to(expectation.device)
        bias_normed = (expectation - true_expectation) / true_expectation
        return bias_normed

    def performance_metrics(self, samples: torch.Tensor, log_w: torch.Tensor,
                            log_q_fn: Optional[LogProbFunc] = None,
                            batch_size: Optional[int] = None) -> Dict:
        bias_normed = self.evaluate_expectation(samples, log_w)
        bias_no_correction = self.evaluate_expectation(samples,
                                                       torch.ones_like(log_w))
        if log_q_fn:
            log_q_test = log_q_fn(self.test_set)
            log_p_test = self.log_prob(self.test_set)
            test_mean_log_prob = torch.mean(log_q_test)
            kl_forward = torch.mean(log_p_test - log_q_test)
            ess_over_p = effective_sample_size_over_p(log_p_test - log_q_test)
            summary_dict = {
                "test_set_mean_log_prob": test_mean_log_prob.cpu().item(),
                "bias_normed": torch.abs(bias_normed).cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item(),
                "ess_over_p": ess_over_p.detach().cpu().item(),
                "kl_forward": kl_forward.detach().cpu().item()
            }
        else:
            summary_dict = {
                "bias_normed": bias_normed.cpu().item(),
                "bias_no_correction": torch.abs(bias_no_correction).cpu().item()
            }
        return summary_dict
