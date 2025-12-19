

from dataclasses import dataclass
from typing import Optional
from .base_config import TrainConfig


@dataclass
class AsymmetricGMMGFConfig(TrainConfig):
    """Configuration for Asymmetric Gaussian Mixture (Gradient Flow) training."""
    # Target-specific parameter: mean offset
    mean_offset: float = 8.0
    use_Ct: bool = False  # Whether to use learned normalization constant Ct

    # Training defaults for GF (matching Funnel GF)
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 400000
    inner_steps: int = 2
    sigma: float = 1.0
    p: float = 2.0  # Generalized Gaussian exponent (p=2 -> Gaussian, p=1 -> Laplace)
    
    # Beta schedule
    beta_min: float = 0.1
    beta_max: float = 20.0
    
    # Evaluation
    eval_interval: int = 200000
    eval_samples: int = 10000
    big_eval_samples: int = 50000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000 # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_AsymGMM/GF"
    run_postfix: str = "run_8.0"
    target_dataset: str = "asymmetric_gmm"


@dataclass
class AsymmetricGMMLearnedConfigX1(TrainConfig):
    """Configuration for Asymmetric Gaussian Mixture (Learned) training."""
    # Target-specific parameter: mean offset (called `m` in some scripts)
    mean_offset: float = 8.0
    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = True  # If True, sample x_t from x1_sampler instead of integrating trajectories. Sets inner_steps=2
    
    # Training defaults for Learned
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 400000
    inner_steps: int = 51
    sigma: float = 1.0
    p: float = 2.0  # Generalized Gaussian exponent 
    
    # Evaluation
    eval_interval: int = 200000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000   # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_AsymGMM/Learned"
    run_postfix: str = "run_8.0"
    target_dataset: str = "asymmetric_gmm"


@dataclass
class AsymmetricGMMLinearConfigX1(TrainConfig):
    """Configuration for Asymmetric Gaussian Mixture (Linear) training."""
    # Target-specific parameter: mean offset (called `m` in some scripts)
    mean_offset: float = 8.0
    use_Ct: bool = False # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = True  # If True, sample x_t from x1_sampler instead of integrating trajectories. Sets inner_steps=2

    # Training defaults for Linear (matching Funnel Linear)
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 500000
    inner_steps: int = 51
    sigma: float = 1.0
    p: float = 2.0  # Generalized Gaussian exponent (p=2 -> Gaussian, p=1 -> Laplace)
    
    # Evaluation
    eval_interval: int = 400000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_AsymGMM/Linear"
    run_postfix: str = "run_8.0"
    target_dataset: str = "asymmetric_gmm"



@dataclass
class AsymmetricGMMLearnedConfig(TrainConfig):
    """Configuration for Asymmetric Gaussian Mixture (Learned) training."""
    # Target-specific parameter: mean offset (called `m` in some scripts)
    mean_offset: float = 8.0
    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories. Sets inner_steps=2

    # Training defaults for Learned (matching Funnel Learned)
    dim: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 10000
    inner_steps: int = 51
    sigma: float = 30.0
    
    # Evaluation
    eval_interval: int = 20000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000   # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_AsymGMM/Learned"
    run_postfix: str = "run_8.0"
    target_dataset: str = "asymmetric_gmm"


@dataclass
class AsymmetricGMMLinearConfig(TrainConfig):
    """Configuration for Asymmetric Gaussian Mixture (Linear) training."""
    # Target-specific parameter: mean offset (called `m` in some scripts)
    mean_offset: float = 8.0
    use_Ct: bool = False # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories. Sets inner_steps=2

    # Training defaults for Linear (matching Funnel Linear)
    dim: int = 2
    batch_size: int = 256
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 40000
    inner_steps: int = 51
    sigma: float = 3.0
    
    # Evaluation
    eval_interval: int = 20000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_AsymGMM/Linear"
    run_postfix: str = "run_8.0"
    target_dataset: str = "asymmetric_gmm"