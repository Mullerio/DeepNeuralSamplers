# trainer.py
import torch
import os
from torch import nn
from tqdm import tqdm
from typing import Callable, Optional, Dict, Any
from fab.utils.utils import save_checkpoint
from models import TrainingModels
from loss import step_gf, step_learned, step_linear
import wandb
from default_factories import get_model_factory
import argparse
from configs.Funnel_configs import FunnelGFConfig, FunnelLearnedConfig, FunnelLinearConfig
from configs.GMM_configs import (
    AsymmetricGMMGFConfig,
    AsymmetricGMMLearnedConfig,
    AsymmetricGMMLearnedConfigX1,
    AsymmetricGMMLinearConfig,
)
from configs.Rings_configs import RingsGFConfig, RingsLearnedConfig, RingsLinearConfig
from fab.target_distributions.funnel import Funnel
from fab.target_distributions.asymmetric_gmm import AsymmetricGMM
from fab.target_distributions.rings import Rings
from evaluation import run_heavy_evaluation, get_eval_callbacks
from compute_action_direct import compute_action_direct
from fab.utils.utils import save_checkpoint
import csv
import copy


def create_evaluate_and_plot_callback(config, training_type):
    """Create evaluation callback for training or standalone evaluation.
    
    Args:
        config: Configuration object with run_dir attribute
        training_type: "gf", "learned", or "linear"
    
    Returns:
        Callable that performs evaluation, checkpoint saving, and action computation
    """
    
    def evaluate_and_plot(step, target, models, device, writer, args):
        # 1. Get callbacks for this target
        target_name = target.__class__.__name__
        plotting_cb, metrics_cb = get_eval_callbacks(target_name)
        
        # Create checkpoint directory for this step
        # For final evaluation (step=None), use "final_{postfix}" naming
        if step is None:
            checkpoint_name = f"final_{config.run_postfix}"
        else:
            checkpoint_name = f"checkpoint_{step}"
        checkpoint_dir = os.path.join(config.run_dir, checkpoint_name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 2. Run heavy evaluation (sampling + plotting)
        # Use final_big_eval samples for final run, normal eval_samples otherwise
        big_eval = getattr(args, 'final_big_eval', False) if step is None else False #TODO: Big eval fix, currently not used
        
        eval_output_dir = os.path.join(checkpoint_dir, "eval_outputs")
        
        # Create a temporary args copy with potentially modified sample counts for final eval
        eval_args = args
        if step is None and hasattr(args, 'final_eval_samples'):
            # For final evaluation, override sample counts if specified
            eval_args = copy.copy(args)
            eval_args.eval_samples = getattr(args, 'final_eval_samples', args.eval_samples)
            eval_args.big_eval_samples = getattr(args, 'final_eval_samples', args.big_eval_samples)
        
        run_heavy_evaluation(
            models=models,
            target=target,
            args=eval_args,
            device=device,
            step=step,
            writer=writer,
            model_type=training_type,
            plotting_callback=plotting_cb,
            metrics_callback=metrics_cb,
            save_dir=eval_output_dir,
            big_eval=None
        )
        
        # 3. Save checkpoint .pt file
        models_dict = {}
        if hasattr(models, 'psi') and models.psi is not None: models_dict['psi'] = models.psi
        if hasattr(models, 'velo') and models.velo is not None: models_dict['velo'] = models.velo
        if hasattr(models, 'schedule') and models.schedule is not None: models_dict['schedule'] = models.schedule
        if hasattr(models, 'Ct') and models.Ct is not None: models_dict['Ct'] = models.Ct
        
        if step is None:
            ckpt_name = f"final_{config.run_postfix}.pt"
        else:
            ckpt_name = f"checkpoint_{step}.pt"
        
        # Add checkpoint args
        checkpoint_args = vars(args) if hasattr(args, "__dict__") else args
        
        save_checkpoint(
            checkpoint_dir,
            ckpt_name,
            models_dict,
            optimizer=None,
            lr_scheduler=None,
            args=checkpoint_args,
            training_type=training_type
        )
        
        action_output_dir = os.path.join(checkpoint_dir, "action_outputs")
        os.makedirs(action_output_dir, exist_ok=True)
        
        # Use custom action sample count for final evaluation if specified
        if step is None and hasattr(args, 'final_action_samples'):
            action_samples = args.final_action_samples
        else:
            action_samples = 2000
        
        action_results = compute_action_direct(
            models=models,
            target=target,
            args=args,
            device=device,
            model_type=training_type,
            num_samples=action_samples,
            num_time_steps=100,
            batch_size=200,
            output_dir=action_output_dir,
            skip_nan=True
        )
        
        # Log to WandB
        if writer and hasattr(writer, 'log'):
            writer.log({
                "Action/mean_action": action_results["total_action"],
                "step": step
            })
            
        # Save to CSV in checkpoint dir
        metrics_path = os.path.join(checkpoint_dir, "action_metrics.csv")
        with open(metrics_path, "w", newline="") as f:
            writer_csv = csv.DictWriter(f, fieldnames=["step", "mean_action"])
            writer_csv.writeheader()
            writer_csv.writerow({"step": step, "mean_action": action_results["total_action"]})
    
    return evaluate_and_plot


class NeuralSamplingTrainer:
    """
    Unified training engine. You supply:
      - args: TrainConfig
      - model_factory: callable(args, device) -> TrainingModels
      - training_type: "gf" | "learned" | "linear"
      - target / gmm as required by step functions
      - evaluate_and_plot: optional callback
    """

    def __init__(self, args, model_factory: Optional[Callable] = None, training_type: str = "linear",
                 target=None, x1_sampler: Optional[Callable] = None,
                 evaluate_and_plot: Optional[Callable] = None, use_wandb: bool = False):
        self.args = args
        self.device = torch.device(args.device)
        self.training_type = training_type
        self.target = target
        self.x1_sampler = x1_sampler
        self.evaluate_and_plot = evaluate_and_plot
        self.use_wandb = use_wandb
        
        # Initialize wandb if requested
        self.writer = None
        if self.use_wandb:
            wandb.init(
                project=getattr(args, "wandb_project", "flow-matching"),
                name=getattr(args, "wandb_run_name", f"{training_type}_{getattr(args, 'sigma', 'default')}"),
                config=vars(args)
            )
            self.writer = wandb

        # create basic models
        if model_factory is None:
            model_factory = get_model_factory(training_type)

        self.models: TrainingModels = model_factory(args, self.device)
        # Collect parameters
        params = []
        for m in (self.models.psi, self.models.velo, self.models.Ct, self.models.schedule):
            if m is not None:
                params += list(m.parameters())

        self.optimizer = torch.optim.Adam(params, lr=args.lr)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=getattr(args, "decay_rate", 0.98))

    def train(self):
        args = self.args
        device = self.device
        progress_bar = tqdm(range(args.ntrain), total=args.ntrain, position=0, leave=True)
        for k in progress_bar:
            self.optimizer.zero_grad()
            if self.training_type == "gf":
                loss, metrics = step_gf(self.models, self.target, args, device, x1_sampler=self.x1_sampler)
                # GF returns loss tensor, call backward once
                loss.backward()
            elif self.training_type == "learned":
                # backward() called inside loop, which is faster
                loss, metrics = step_learned(self.models, self.target, args, device, x1_sampler=self.x1_sampler)
            elif self.training_type == "linear":
                # backward() called inside loop, which is faster
                loss, metrics = step_linear(self.models, self.target, args, device, x1_sampler=self.x1_sampler)
            else:
                raise ValueError(f"Unknown training_type {self.training_type}")


            # safe default clipping
            for m in (self.models.psi, self.models.velo, self.models.Ct):
                if m is not None:
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 100.0)

            self.optimizer.step()

            # LR schedule stepping rules, TODO: figure good ones out?
            if self.training_type == "learned":
                if k % 250 == 0:
                    self.lr_scheduler.step()
            elif self.training_type == "linear":
                if k % 50 == 0:
                    self.lr_scheduler.step()
            else:
                if k % 1000 == 0:
                    self.lr_scheduler.step()

            if self.use_wandb:
                wandb.log({
                    "loss": metrics.get("last_loss", float(loss.detach())),
                    "lr": self.lr_scheduler.get_last_lr()[0] if hasattr(self.lr_scheduler, "get_last_lr") else 0.0,
                    "step": k
                })

            if (k + 1) % getattr(args, "eval_interval", 200) == 0 and self.evaluate_and_plot is not None:
                self.evaluate_and_plot(k, self.target, self.models, device, self.writer, args)

            progress_bar.set_description(f"Loss={metrics.get('last_loss', float(loss.detach())):.3f}")

        # final evaluation + plotting instead of simple checkpoint save
        if self.evaluate_and_plot is not None:
            # perform final evaluation (step=None indicates final)
            self.evaluate_and_plot(None, self.target, self.models, device, self.writer, args)
            fname = None
        else:
            # fallback: save checkpoint into save_dir
            save_dir = getattr(args, "save_dir", None) or f"nets_generic_{self.training_type}"
            os.makedirs(save_dir, exist_ok=True)
            ckpt_name = f"checkpoint_{self.training_type}_{getattr(args, 'sigma', '')}.pt"
            models_dict = {k: v for k, v in self.models.__dict__.items() if k in ("psi", "velo", "Ct", "schedule")}
            fname = save_checkpoint(save_dir, ckpt_name, models_dict, optimizer=self.optimizer, lr_scheduler=self.lr_scheduler, args=args, training_type=self.training_type)
    
        if self.use_wandb:
            wandb.finish()

        return fname


def main():
    """CLI for training neural sampling models."""
    parser = argparse.ArgumentParser(description="Train neural sampling models")
    parser.add_argument("--config", type=str, required=True, 
                       choices=["funnel_gf", "funnel_learned", "funnel_linear", 
                               "asymmetric_gmm_gf", "asymmetric_gmm_learned", "asymmetric_gmm_linear",
                               "rings_gf", "rings_learned", "rings_linear"],
                       help="Configuration to use")
    parser.add_argument("--dim", type=int, help="Override dimension")
    parser.add_argument("--sigma", type=float, help="Override initial sigma (std dev)")
    parser.add_argument("--sigma_funnel", type=float, help="Override funnel sigma")
    parser.add_argument("--ntrain", type=int, help="Override number of training steps")
    parser.add_argument("--batch_size", type=int, help="Override batch size")
    parser.add_argument("--lr", type=float, help="Override learning rate")
    parser.add_argument("--device", type=str, help="Override device (cuda/cpu)")
    parser.add_argument("--mean_offset", type=float, help="Override mean offset (m) for Asymmetric GMM")
    parser.add_argument("--num_modes", type=int, help="Override number of modes/rings for Rings")
    parser.add_argument("--radius", type=float, help="Override radius for Rings")
    parser.add_argument("--sigma_rings", type=float, help="Override sigma for Rings")
    parser.add_argument("--p", type=float, help="Override generalized Gaussian p parameter (p=2 -> Gaussian, p=1 -> Laplace)")
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="flow-matching", help="W&B project name")
    parser.add_argument("--wandb_run_name", type=str, help="W&B run name")
    parser.add_argument("--save_dir", type=str, help="Override save directory")
    parser.add_argument("--run_dir", type=str, help="Override run directory")
    parser.add_argument("--final_eval_samples", type=int, help="Override number of samples for final evaluation")
    parser.add_argument("--final_action_samples", type=int, help="Override number of samples for final action computation")
    parser.add_argument("--divide_by_two", type=str, choices=["True", "False"], help="For Funnel: use exp(y)/2 variance (only for configs that support it)")
    cli_args = parser.parse_args()
    
    # Load base config
    config_map = {
        "funnel_gf": FunnelGFConfig,
        "funnel_learned": FunnelLearnedConfig,
        "funnel_linear": FunnelLinearConfig,
        "asymmetric_gmm_gf": AsymmetricGMMGFConfig,
        "asymmetric_gmm_learned": AsymmetricGMMLearnedConfig,
        "asymmetric_gmm_linear": AsymmetricGMMLinearConfig,
        "rings_gf": RingsGFConfig,
        "rings_learned": RingsLearnedConfig,
        "rings_linear": RingsLinearConfig,
    }
    
    ConfigClass = config_map[cli_args.config]
    config = ConfigClass()
    
    if cli_args.dim is not None:
        config.dim = cli_args.dim
    if cli_args.sigma is not None:
        config.sigma = cli_args.sigma
    if cli_args.sigma_funnel is not None:
        config.sigma_funnel = cli_args.sigma_funnel
    if cli_args.mean_offset is not None and hasattr(config, "mean_offset"):
        config.mean_offset = cli_args.mean_offset
    if cli_args.num_modes is not None and hasattr(config, "num_modes"):
        config.num_modes = cli_args.num_modes
    if cli_args.radius is not None and hasattr(config, "radius"):
        config.radius = cli_args.radius
    if cli_args.sigma_rings is not None and hasattr(config, "sigma_rings"):
        config.sigma_rings = cli_args.sigma_rings
    if cli_args.p is not None and hasattr(config, "p"):
        config.p = cli_args.p
    if cli_args.ntrain is not None:
        config.ntrain = cli_args.ntrain
    if cli_args.batch_size is not None:
        config.batch_size = cli_args.batch_size
    if cli_args.lr is not None:
        config.lr = cli_args.lr
    if cli_args.device is not None:
        config.device = cli_args.device
    if cli_args.save_dir is not None:
        config.save_dir = cli_args.save_dir
    if cli_args.run_dir is not None:
        config.run_dir = cli_args.run_dir
    if cli_args.wandb_project:
        config.wandb_project = cli_args.wandb_project
    if cli_args.wandb_run_name:
        config.wandb_run_name = cli_args.wandb_run_name
    if cli_args.divide_by_two is not None:
        config.divide_by_two = cli_args.divide_by_two == "True"
    
    # Update run_postfix based on actual parameter values
    if cli_args.config.startswith("funnel"):
        sigma_funnel_val = getattr(config, 'sigma_funnel', 3.0)
        sigma_val = getattr(config, 'sigma', 1.0)
        divide_var = getattr(config, 'divide_by_two', True)
        p_val = getattr(config, 'p', 2.0)
        config.run_postfix = f"run_{sigma_funnel_val}_{divide_var}_sig{sigma_val}_p{p_val}"
    elif cli_args.config.startswith("asymmetric_gmm"):
        mean_offset_val = getattr(config, 'mean_offset', 8.0)
        sigma_val = getattr(config, 'sigma', 1.0)
        p_val = getattr(config, 'p', 2.0)
        use_x1_val = getattr(config, 'use_x1_sampler', False)
        x1_str = "x1True" if use_x1_val else "x1False"
        config.run_postfix = f"run_{mean_offset_val}_sig{sigma_val}_p{p_val}_{x1_str}"
    elif cli_args.config.startswith("rings"):
        num_modes_val = getattr(config, 'num_modes', 4)
        radius_val = getattr(config, 'radius', 1.0)
        sigma_rings_val = getattr(config, 'sigma_rings', 0.15)
        sigma_val = getattr(config, 'sigma', 1.0)
        p_val = getattr(config, 'p', 2.0)
        use_x1_val = getattr(config, 'use_x1_sampler', False)
        x1_str = "x1True" if use_x1_val else "x1False"
        config.run_postfix = f"run_{num_modes_val}_{radius_val}_{sigma_rings_val}_sig{sigma_val}_p{p_val}_{x1_str}"
    
    # Ensure run_dir is a valid string so later os.path.join calls don't fail
    # Prefer explicit CLI override, then config's run_root/postfix, then save_dir, then cwd
    if getattr(config, 'run_dir', None) is None:
        run_root = getattr(config, 'run_root', None)
        run_postfix = getattr(config, 'run_postfix', None)
        if run_root and run_postfix:
            config.run_dir = os.path.join(run_root, run_postfix)
        else:
            config.run_dir = run_root or getattr(config, 'save_dir', None) or os.getcwd()
    training_type = cli_args.config.lower().split("_")[-1]  # "gf", "learned", or "linear"

    # Set up run directory
    os.makedirs(config.run_dir, exist_ok=True)

            
    # Create target distribution (target-agnostic)
    x1_sampler = None
    if cli_args.config.startswith("asymmetric_gmm"):
        def x1_sampler(n, dim, device):
            box_min = -config.mean_offset - 10.0
            box_max = 4.0 + 10.0
            box_width = box_max - box_min
        
            return torch.rand((n, dim), device=device) * box_width + box_min    
        
        target = AsymmetricGMM(
            dim=config.dim,
            mean_offset=getattr(config, "mean_offset", 8.0),
            use_gpu=(config.device == "cuda")
        )
        
    elif cli_args.config.startswith("funnel"):
        divide_variance_by_two = config.divide_by_two
        target = Funnel(
            dim=config.dim,
            sigma=config.sigma_funnel,
            use_gpu=(config.device == "cuda"),
            divide_variance_by_two=divide_variance_by_two
        )
        
        # For Funnel, use Gaussian * sigma_funnel as x1_sampler, box is a bit weird here due to low probability regions
        def x1_sampler(n, dim, device):
            samples = torch.randn(n, dim, device=device)
            return samples 
    
    elif cli_args.config.startswith("rings"):
        target = Rings(
            num_modes=getattr(config, "num_modes", 4),
            radius=getattr(config, "radius", 1.0),
            sigma=getattr(config, "sigma_rings", 0.15),
            use_gpu=(config.device == "cuda")
        )
        
        # For Rings, x1_sampler is a box around the rings
        def x1_sampler(n, dim, device):
            box_min = target.x_min - 2 * target.sigma
            box_max = target.x_max + 2* target.sigma
            box_width = box_max - box_min
            return torch.rand((n, dim), device=device) * box_width + box_min
    
    print(f"{'='*70}")
    print(f"Training Configuration: {cli_args.config}")
    print(f"{'='*70}")
    print(f"Training type: {training_type}")
    print(f"Dimension: {config.dim}")
    if cli_args.config.startswith("asymmetric_gmm"):
        print(f"AsymmetricGMM mean_offset: {getattr(config, 'mean_offset', 8.0)}")
    elif cli_args.config.startswith("funnel"):
        print(f"Funnel sigma: {config.sigma_funnel}")
    elif cli_args.config.startswith("rings"):
        print(f"Rings num_modes: {getattr(config, 'num_modes', 4)}")
        print(f"Rings radius: {getattr(config, 'radius', 1.0)}")
        print(f"Rings sigma: {getattr(config, 'sigma_rings', 0.15)}")
    print(f"Training steps: {config.ntrain}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.lr}")
    print(f"Device: {config.device}")
    print(f"W&B enabled: {cli_args.use_wandb}")
    print(f"{'='*70}\n")
    
    # Create evaluation callback
    evaluate_and_plot = create_evaluate_and_plot_callback(config, training_type)
    
    # Create trainer
    trainer = NeuralSamplingTrainer(
        args=config,
        training_type=training_type,
        target=target,
        x1_sampler=x1_sampler,
        use_wandb=cli_args.use_wandb,
        evaluate_and_plot=evaluate_and_plot
    )
    
    # Train
    checkpoint_path = trainer.train()
    print(f"\n{'='*70}")
    print(f"Training complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()


