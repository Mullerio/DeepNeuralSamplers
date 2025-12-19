# Rings_configs.py
from dataclasses import dataclass
from typing import Optional
from .base_config import TrainConfig


@dataclass
class RingsGFConfig(TrainConfig):
    """Configuration for Rings distribution with Gradient Flow training."""
    # Target-specific parameters
    num_modes: int = 4  # Number of concentric rings
    radius: float = 1.0  # Radius of the smallest ring
    sigma_rings: float = 0.15  # Width of each ring
    use_Ct: bool = False  # Whether to use learned normalization constant Ct

    # Training defaults for GF
    dim: int = 2  # Rings are always 2D
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 400000
    inner_steps: int = 2
    sigma: float = 1.0  # Sigma for initial gaussian
    p: float = 2.0  
    
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
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 10000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_Rings/GF"
    run_postfix: str = "run_4_1.0_0.15"
    target_dataset: str = "rings"


@dataclass
class RingsLearnedConfig(TrainConfig):
    """Configuration for Rings distribution with Learned training."""
    # Target-specific parameters
    num_modes: int = 4  # Number of concentric rings
    radius: float = 1.0  # Radius of the smallest ring
    sigma_rings: float = 0.15  # Width of each ring
    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories
    
    # Training defaults for Learned
    dim: int = 2  # Rings are always 2D
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 30000
    inner_steps: int = 51
    sigma: float = 20.0  # Sigma for initial gaussian
    p: float = 2.0  # Generalized Gaussian exponent
    
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
    run_root: str = "runs_Rings/Learned"
    run_postfix: str = "run_4_1.0_0.15"
    target_dataset: str = "rings"


@dataclass
class RingsLinearConfig(TrainConfig):
    """Configuration for Rings distribution with Linear training."""
    # Target-specific parameters
    num_modes: int = 4  # Number of concentric rings
    radius: float = 1.0  # Radius of the smallest ring
    sigma_rings: float = 0.15  # Width of each ring
    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories

    # Training defaults for Linear
    dim: int = 2  # Rings are always 2D
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 30000
    inner_steps: int = 51
    sigma: float = 20.0  # Sigma for initial gaussian
    p: float = 2.0  # Generalized Gaussian exponent
    
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
    run_root: str = "runs_Rings/Linear"
    run_postfix: str = "run_4_1.0_0.15"
    target_dataset: str = "rings"
