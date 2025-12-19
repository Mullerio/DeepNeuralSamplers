# utils.py
import os
import torch
from typing import Any, Dict, Optional
from datetime import datetime

def save_checkpoint(save_dir: str, name: str, models: Dict[str, Optional[torch.nn.Module]],
                    optimizer: Optional[torch.optim.Optimizer] = None,
                    lr_scheduler: Optional[Any] = None,
                    args: Optional[Any] = None,
                    training_type: Optional[str] = None):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint = {
        "models": {k: (v.state_dict() if v is not None else None) for k, v in models.items()},
        "optimizers": {"optimizer": optimizer.state_dict()} if optimizer is not None else {},
        "lr_schedulers": {"lr_scheduler": lr_scheduler.state_dict()} if lr_scheduler is not None else {},
        "args": vars(args) if hasattr(args, "__dict__") else args,
        "training_type": training_type,
        "saved_at": datetime.utcnow().isoformat() + "Z"
    }
    fname = os.path.join(save_dir, name)
    torch.save(checkpoint, fname)
    return fname

