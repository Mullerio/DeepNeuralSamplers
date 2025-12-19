from dataclasses import dataclass
from typing import Optional

@dataclass
class TrainConfig:
	dim: int
	batch_size: int
	ntrain: int
	inner_steps: int
	sigma: float

	lr: float = 1e-4
	decay_rate: float = 0.98
	eval_interval: int = 200

	beta_min: float = 0.1
	beta_max: float = 20.0

	run_dir: Optional[str] = None
	save_dir: Optional[str] = None
	run_postfix: Optional[str] = None

	ode_atol: float = 1e-4
	ode_rtol: float = 1e-4

	device: str = "cuda"