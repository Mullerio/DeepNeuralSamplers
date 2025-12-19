"""
Default model factory functions for GF, Learned, and Linear training.
"""

from models import MLP3, MLP4, MLP2, TrainingModels


def create_gf_models(args, device):
    """
    Create models for gradient flow (GF) training.
    
    Architecture:
    - psi: MLP4 (5 layers, w=512) for interpolation network
    - schedule: MLP3 (6 layers, w=128) for schedule network
    - Ct: optional normalization constant (controlled by args.use_Ct, default False)
    - For gf, velo is computed from psi 
    """
    psi = MLP4(dim=args.dim, out_dim=1, time_varying=True, w=512).to(device)
    schedule = MLP3(dim=1, out_dim=1, time_varying=False, w=128).to(device)
    Ct = MLP3(dim=1, out_dim=1, time_varying=False, w=128).to(device) if getattr(args, 'use_Ct', False) else None
    
    return TrainingModels(psi=psi, velo=None, Ct=Ct, schedule=schedule)


def create_learned_models(args, device):
    """
    Create models for learned training.
    
    Architecture:
    - psi: MLP4 (5 layers, w=512) for interpolation network
    - velo: MLP4 (5 layers, w=512) for velocity network
    - Ct: MLP3 (6 layers, w=256) for normalization constant (optional, controlled by args.use_Ct, default True)
    - No schedule needed for learned
    """
    psi = MLP2(dim=args.dim, out_dim=1, time_varying=True, w=512).to(device)
    velo = MLP2(dim=args.dim, out_dim=args.dim, time_varying=True, w=512).to(device)
    Ct = MLP3(dim=1, out_dim=1, time_varying=False, w=256).to(device) if getattr(args, 'use_Ct', True) else None
    
    return TrainingModels(psi=psi, velo=velo, Ct=Ct, schedule=None)


def create_linear_models(args, device):
    """
    Create models for linear training.
    
    Architecture:
    - velo: MLP4 (5 layers, w=512) for velocity network
    - Ct: MLP4 (5 layers, w=512) for normalization constant (optional, controlled by args.use_Ct, default True)
    - No psi or schedule needed for linear
    """
    velo = MLP2(dim=args.dim, out_dim=args.dim, time_varying=True, w=512).to(device)
    Ct = MLP3(dim=1, out_dim=1, time_varying=False, w=512).to(device) if getattr(args, 'use_Ct', True) else None
    
    return TrainingModels(psi=None, velo=velo, Ct=Ct, schedule=None)


def get_model_factory(training_type: str):
    factories = {
        "gf": create_gf_models,
        "learned": create_learned_models,
        "linear": create_linear_models,
    }

    return factories[training_type]
