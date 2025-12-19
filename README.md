# DeepNeuralSamplers

This is a implementation of [https://arxiv.org/abs/2410.03282](https://arxiv.org/abs/2410.03282) by Chemseddine et. al. and [https://arxiv.org/abs/2410.03282](https://arxiv.org/abs/2301.07388) by Máté et al. to learn samplers from unnormalized densities. 

We support the three methods presented:
- **GF** (Gradient flow from [https://arxiv.org/abs/2410.03282](https://arxiv.org/abs/2410.03282))
- **Learned** 
- **Linear** 

## Quick Start

First, install dependencies:

```bash
uv sync
```

or withotu uv 

```bash
pip install -e .
```

### Train a single variant:

```bash
uv run python -m src.trainer --config rings_gf --num_modes 4 --radius 1.0 --sigma_rings 0.15
```


### Run all three variants in parallel:

```bash
uv run python -m src.run_all_variants --target funnel --sigma_funnel 3.0
```

## Configuration

All configs are in `src/configs/`. Key parameters:
- `--config`: Specific config to use (e.g., funnel_gf, asymmetric_gmm_learned, rings_linear)
- `--sigma_funnel`: Funnel sigma parameter
- `--mean_offset`: GMM mean offset
- `--num_modes`: Number of modes for Rings
- `--radius`: Radius for Rings
- `--sigma_rings`: Sigma for Rings
- `--ntrain`: Number of training steps
- `--lr`: Learning rate

## Structure

```
src/
├── trainer.py              # Main training script
├── models.py               # Neural Networks
├── loss.py                 # Loss functions
├── evaluation.py           # Plotting and Eval
├── compute_action_direct.py # Computation of Action
├── run_all_variants.py     # Run all three variants in parallel
├── default_factories.py     # Model factory functions
├── flow_wrappers.py        # Flow utilities
├── configs/                # Configuration classes
│   ├── base_config.py
│   ├── Funnel_configs.py
│   ├── GMM_configs.py
│   └── Rings_configs.py
└── fab/                    
    ├── utils/              
    └── target_distributions/  # Target distributions
```



### TODO:

This codebase is still WIP, i.e. the wandb integration is not fully updated and eval in general needs some work (big eval not used consistently). 
