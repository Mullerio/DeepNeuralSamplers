# config.py
from dataclasses import dataclass
from typing import Optional
from .base_config import TrainConfig


@dataclass
class FunnelGFConfig(TrainConfig):
    """Configuration for Funnel distribution with Gradient Flow training."""
    # Target distribution
    sigma_funnel: float = 1.0  # Sigma for y in Funnel distribution
    divide_by_two: bool = True  # Use exp(y)/2 variance (set False for exp(y))
    use_Ct: bool = False  # Whether to use learned normalization constant Ct

    
    # Training defaults for GF
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 1000000
    inner_steps: int = 2
    sigma: float = 1.0 #sigma for initial gaussian, THIS IS NOT THE SIGMA FOR THE FUNNEL it should stay 1.0
    
    # Beta schedule
    beta_min: float = 0.1
    beta_max: float = 20.0
        
    # Evaluation
    eval_interval: int = 200000
    eval_samples: int = 10000
    big_eval_samples: int = 100000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 100000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_Funnel/GF"
    run_postfix: str = "run_3.0_False"
    target_dataset: str = "funnel"


@dataclass
class FunnelLearnedConfig(TrainConfig):
    """Configuration for Funnel distribution with Learned (non-linear) training."""
    # Target distribution
    sigma_funnel: float = 1.0  # Sigma for y in Funnel distribution
    divide_by_two: bool = True  # Use exp(y)/2 variance (set False for exp(y))

    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories
    
    # Training defaults for Learned
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 20000
    inner_steps: int = 51
    sigma: float = 1.0
    
    # Evaluation
    eval_interval: int = 4000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 1000000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_Funnel/Learned"
    run_postfix: str = "run_3.0_False"
    target_dataset: str = "funnel"


@dataclass
class FunnelLinearConfig(TrainConfig):
    """Configuration for Funnel distribution with Linear training."""
    # Target distribution
    sigma_funnel: float = 1.0  # Sigma for y in Funnel distribution
    divide_by_two: bool = True  # Use exp(y)/2 variance (set False for exp(y))

    use_Ct: bool = False  # Whether to use learned normalization constant Ct
    use_x1_sampler: bool = False  # If True, sample x_t from x1_sampler instead of integrating trajectories

    
    # Training defaults for Linear
    dim: int = 2
    batch_size: int = 4096
    lr: float = 1e-3
    decay_rate: float = 0.98
    ntrain: int = 20000
    inner_steps: int = 51
    sigma: float = 1.0
    
    # Evaluation
    eval_interval: int = 10000
    eval_samples: int = 10000
    big_eval_samples: int = 10000
    eval_batch: int = 5000
    num_steps_eval: int = 100
    final_big_eval: bool = False
    final_eval_samples: int = 1000000  # Samples for final evaluation
    final_eval_plotting: int = 1000000  # Samples to use for plotting (if smaller than final_eval_samples)
    final_action_samples: int = 1000  # Samples for final action computation
    
    # Directories
    run_root: str = "runs_Funnel/Linear"
    run_postfix: str = "run_3.0_False"
    target_dataset: str = "funnel"


