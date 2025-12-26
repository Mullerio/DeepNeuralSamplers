import torch
import os
import sys
import numpy as np
from torchdiffeq import odeint
from eval_utils import cnf_sample
from loss import TorchWrapper, TorchWrapperGF, beta
import traceback
import subprocess
from fab.utils.utils import save_checkpoint
import matplotlib.pyplot as plt
import torch.nn.functional as F
from geomloss import SamplesLoss
from fab.utils.funnel_plotting import plot_funnel_2d, plot_latent_colored_by_target_norm
from fab.utils.plotting import plot_contours_mixture
import fab.utils.funnel_eval as funnel_eval
import csv



def run_heavy_evaluation(
    models,
    target,
    args,
    device,
    step=None,
    writer=None,
    model_type="linear",
    plotting_callback=None,
    metrics_callback=None,
    save_dir=None,
    big_eval=False
):
    """
    Base version of run_heavy_evaluation_with_backward_traj.
    
    Args:
        models (TrainingModels or dict): Models (psi, velo, schedule, etc.).
        target (TargetDistribution): Target distribution.
        args (TrainConfig): Configuration arguments.
        device (torch.device): Device.
        step (int, optional): Current training step. Defaults to None.
        writer (SummaryWriter, optional): TensorBoard/WandB writer.
        model_type (str): "gf", "learned", or "linear".
        plotting_callback (callable, optional): Function(x_gen, eps_kept, step, save_dir, big_eval).
        metrics_callback (callable, optional): Function(x_gen, eps_kept, step, save_dir, big_eval) -> dict.
        save_dir (str, optional): Directory to save outputs.
        big_eval (bool): Whether this is a "big" evaluation (more samples).
    """
    step_str = str(step) if step is not None else "final"
    print(f"\n{'='*60}")
    print(f"Running {'BIG ' if big_eval else ''}evaluation at step {step_str}")
    print(f"{'='*60}")
    
    if save_dir is None:
        save_dir = os.path.join(args.run_dir, "eval_outputs")
    os.makedirs(save_dir, exist_ok=True)

    # 1. Create Flow Wrapper
    # Use wrappers from loss.py as requested
    
    if model_type in ["learned", "linear"]:
        if hasattr(models, 'velo'):
            velo = models.velo
        elif isinstance(models, dict) and 'velo' in models:
            velo = models['velo']
        else:
            raise ValueError(f"Model type {model_type} requires 'velo' model")
            
        # Use TorchWrapper from loss.py
        wrapper_v = TorchWrapper(velo)
        wrapper = cnf_sample(wrapper_v)
        
        # Integration settings for Learned/Linear (Forward 0->1)
        num_steps = getattr(args, 'num_steps_eval', 100)
        t_span = torch.linspace(0, 1, num_steps).to(device)
        ode_kwargs = {
            "method": "rk4"
        }
        
    elif model_type == "gf":
        if hasattr(models, 'psi'):
            psi = models.psi
            schedule = models.schedule
        elif isinstance(models, dict):
            psi = models['psi']
            schedule = models['schedule']
            
        beta_min = getattr(args, 'beta_min', 0.1)
        beta_max = getattr(args, 'beta_max', 20.0)
        beta_fn = lambda t: beta(t, beta_min, beta_max)
        
        # Use TorchWrapperGF from loss.py
        wrapper_v = TorchWrapperGF(psi, schedule, target, beta_fn)
        wrapper = cnf_sample(wrapper_v)
        
        # Integration settings for GF (Backward 0.999->0.001)
        num_steps = getattr(args, 'num_steps_eval', 100)
        t_span = torch.linspace(0.999, 0.001, num_steps, device=device)
        ode_kwargs = {
            "method": "rk4"
        }
        """ode_kwargs = {
            "atol": 1e-6, 
            "rtol": 1e-6, 
            "method": "dopri5",
        }
        """

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 2. Generate Samples
    total_samples = int(args.big_eval_samples if big_eval else args.eval_samples)
    if step is None and hasattr(args, 'final_eval_samples'):
        total_samples = int(args.final_eval_samples)
        
    total_samples = 1000000
    batch_size = int(getattr(args, 'eval_batch', 1024))
    
    generated_samples = []
    kept_eps = []
    log_p_list = []  # log prob under target
    log_q_list = []  # log prob under base (initial)
    weights = []
    
    # Lists to store per-batch statistics
    nll_list = []  # NLL per batch
    energy_distances = []  # Energy distance per batch
    ess_list = []  # ESS per batch
    
    # Store full trajectories for visualization (up to 1000 samples)
    full_trajectories = []  # Will store (num_timesteps, dim) trajectories
    trajectory_t_span = t_span.cpu().numpy() if step is None else None  # Only store times for final eval
    
    seen = 0
    num_loops = (total_samples + batch_size - 1) // batch_size
    
    print(f"  Generating {total_samples} samples...")
    
    for loop_idx in range(num_loops):
        current_bs = min(batch_size, total_samples - seen)
        if current_bs <= 0:
            break
            
        print(f"  Batch {loop_idx+1}/{num_loops} with batch size {current_bs}")
        try:
            with torch.no_grad():
                start = torch.randn((current_bs, args.dim), device=device) * args.sigma
                
                # Integrate
                traj, logs = odeint(
                    wrapper,
                    (start, torch.zeros(current_bs, 1).type(torch.float32).to(device)),
                    t_span,
                    **ode_kwargs
                )
                
                # Extract final samples
                x_gen = traj[-1].detach()
                logs = logs[-1].detach()
                
                # Compute log probabilities
                log_p = target.log_prob(x_gen).detach()
                log_q = (-0.5 * torch.sum(start**2, dim=1) / (args.sigma**2)).detach()
                
                # Compute NLL for this batch
                batch_nll = (-log_p).mean().item()
                nll_list.append(batch_nll)
                
                seen += current_bs
            
    
                true_samples_batch = target.sample((current_bs,)).to(device)
                energy_loss = SamplesLoss(loss="energy", backend="auto")
                ed = energy_loss(true_samples_batch, x_gen).item()
                
                if not np.isnan(ed) and ed >= 0:
                    energy_distances.append(ed)
                    print(f"    Batch NLL: {batch_nll:.6f}, Energy Distance: {ed:.6f}")
                else:
                    print(f"    Batch NLL: {batch_nll:.6f}, Energy Distance: NaN")
    

                log_flow_weights = log_q - logs.squeeze()
                log_weights = log_p - log_flow_weights
                normalized_weights = F.softmax(log_weights, dim=0)
                batch_ess = (1.0 / torch.sum(normalized_weights ** 2) / normalized_weights.shape[0]).item()
                ess_list.append(batch_ess)
         
                    
                weights.append(logs.cpu())
                generated_samples.append(x_gen.cpu())
                kept_eps.append(start.cpu())
                log_p_list.append(log_p.cpu())
                log_q_list.append(log_q.cpu())
                
                # Store full trajectories for final plotting (only up to 1000 for visual clarity)
                if step is None and len(full_trajectories) < 1000:
                    # traj shape: (num_timesteps, batch_size, dim)
                    traj_cpu = traj.detach().cpu().numpy()
                    num_to_store = min(current_bs, 1000 - len(full_trajectories))
                    for i in range(num_to_store):
                        # Store (num_timesteps, dim) trajectory
                        full_trajectories.append(traj_cpu[:, i, :])
                
        except Exception as e:
            print(f"  Error in batch {loop_idx}: {e}")
            traceback.print_exc()
            continue
    
    if not generated_samples:
        print("  Failed to generate any samples!")
        return None
        
    x_gen = torch.cat(generated_samples, dim=0)
    eps_kept = torch.cat(kept_eps, dim=0)
    
    print(f"  Generated {x_gen.shape[0]} samples")
    
    # Compute statistics from batch lists
    mean_nll = np.mean(nll_list) if nll_list else float('nan')
    std_nll = np.std(nll_list) if len(nll_list) > 1 else 0.0
    print(f"  Mean NLL: {mean_nll:.6f} +/- {std_nll:.6f}")
    
    if energy_distances:
        mean_ed = np.mean(energy_distances)
        std_ed = np.std(energy_distances) if len(energy_distances) > 1 else 0.0
        print(f"  Energy Distance: {mean_ed:.6f} +/- {std_ed:.6f}")
        energy_distance = mean_ed
        energy_distance_std = std_ed
    else:
        print(f"  WARNING: No valid energy distance values computed")
        energy_distance = float('nan')
        energy_distance_std = float('nan')
    
    if ess_list:
        mean_ess = np.mean(ess_list)
        std_ess = np.std(ess_list) if len(ess_list) > 1 else 0.0
        print(f"  ESS: {mean_ess:.6f} +/- {std_ess:.6f}")
        ess = mean_ess
        ess_std = std_ess
    else:
        print(f"  WARNING: No valid ESS values computed")
        ess = float('nan')
        ess_std = float('nan')
    
    # 3. Plotting 
    x_gen_plot = x_gen
    eps_kept_plot = eps_kept
    
    if step is None:  # Only for final evaluation
        final_eval_plotting = getattr(args, 'final_eval_plotting', None)
        if final_eval_plotting is not None and final_eval_plotting < x_gen.shape[0]:
            x_gen_plot = x_gen[:final_eval_plotting]
            eps_kept_plot = eps_kept[:final_eval_plotting]
    
    if plotting_callback:
        print("  Running plotting callback...")
        try:
            plotting_callback(x_gen_plot, eps_kept_plot, step, save_dir, big_eval, target, 
                            models=models, model_type=model_type, args=args, device=device,
                            full_trajectories=full_trajectories if step is None else None,
                            trajectory_t_span=trajectory_t_span)
        except Exception as e:
            print(f"  Plotting failed: {e}")
            traceback.print_exc()

    all_metrics = {
        "mean_nll": mean_nll,
        "std_nll": std_nll,
        "energy_distance": energy_distance,
        "energy_distance_std": energy_distance_std,
        "ess": ess,
        "ess_std": ess_std
    }
    
    if metrics_callback:
        print("  Running metrics callback...")
        try:
            extra_metrics = metrics_callback(x_gen, eps_kept, step, save_dir, big_eval, target)
            if extra_metrics:
                all_metrics.update(extra_metrics)
        except Exception as e:
            print(f"  Metrics computation failed: {e}")
            traceback.print_exc()
            
    # 5. Logging
    # Save to CSV
    metrics_path = os.path.join(save_dir, "metrics.csv")
    write_header = not os.path.exists(metrics_path)
    with open(metrics_path, "a", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["step"] + list(all_metrics.keys()))
        if write_header:
            writer_csv.writeheader()
        
        row = {"step": step if step is not None else "final", **all_metrics}
        writer_csv.writerow(row)
    
    # Log to Writer (WandB only)
    if writer:
        prefix = "BigEval" if big_eval else "Eval"
        if hasattr(writer, 'log'):
             log_dict = {f"{prefix}/{key}": val for key, val in all_metrics.items()}
             if step is not None:
                 log_dict['step'] = step
             writer.log(log_dict)

    print(f"  Metrics: {all_metrics}")
    return all_metrics

def plot_velocity_field(models, model_type, args, device, target=None, timesteps=None, 
                        grid_range=(-15, 15), grid_points=35, save_dir=None, step=None):
    """
    Plot velocity field(s) at specified timesteps on a 2D grid.
    
    Arrows all have the same length (normalized), with color indicating the actual velocity magnitude.
    A colorbar on each plot shows the mapping from color to velocity magnitude (speed).
    
    Args:
        models: TrainingModels or dict with velo/psi/schedule models
        model_type (str): "gf", "learned", or "linear"
        args: Configuration with beta_min, beta_max, sigma, etc.
        device: torch device
        target: Target distribution (needed for GF, unused for learned/linear)
        timesteps (list, list of lists, or float): 
            - Single float: plot that timestep
            - List of floats: plot all in one figure
            - List of lists: create separate plots for each list, plus combined plot
            If None, plots [[0.01, 0.10, 0.20,0.25, 0.5, 0.75, 0.8, 0.90, 0.99]]
        grid_range (tuple): (min, max) for grid bounds
        grid_points (int): Number of grid points per dimension
        save_dir (str): Directory to save plots
        step (int): Training step (for filename)
    """
    if timesteps is None:
        timesteps = [[0.01, 0.10, 0.20,0.25, 0.5, 0.75, 0.8, 0.90, 0.99]]
    if timesteps is None and step is None: 
        timesteps = [[0.01], [0.10], [0.20], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [0.99], [0.999]]
    elif isinstance(timesteps, (int, float)):
        timesteps = [[timesteps]]
    elif isinstance(timesteps, list):
        # Check if it's a list of lists or list of floats
        if timesteps and isinstance(timesteps[0], (list, tuple)):
            # Already a list of lists
            pass
        else:
            # List of floats, wrap it
            timesteps = [timesteps]
    
    if save_dir is None:
        save_dir = "velocity_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create grid
    x_range = np.linspace(grid_range[0], grid_range[1], grid_points)
    y_range = np.linspace(grid_range[0], grid_range[1], grid_points)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points_xy = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (N, 2)
    grid_points_tensor = torch.from_numpy(grid_points_xy).float().to(device)
    
    # Get wrapper
    if model_type in ["learned", "linear"]:
        velo = models.velo if hasattr(models, 'velo') else models['velo']
        wrapper = TorchWrapper(velo)
    elif model_type == "gf":
        psi = models.psi if hasattr(models, 'psi') else models['psi']
        schedule = models.schedule if hasattr(models, 'schedule') else models['schedule']
        beta_min = getattr(args, 'beta_min', 0.1)
        beta_max = getattr(args, 'beta_max', 20.0)
        beta_fn = lambda t: beta(t, beta_min, beta_max)
        wrapper = TorchWrapperGF(psi, schedule, target, beta_fn)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Precompute target log probability grid once
    target_log_prob_grid = None
    if target is not None:
        try:
            target_device = target.device if hasattr(target, 'device') else 'cpu'
            target_cpu = target.to('cpu')
            
            with torch.no_grad():
                grid_tensor_cpu = torch.from_numpy(grid_points_xy).float()
                # Pad to full dimension if needed
                if grid_tensor_cpu.shape[1] < target_cpu.dim:
                    padding = torch.zeros(grid_tensor_cpu.shape[0], target_cpu.dim - 2)
                    grid_tensor_cpu = torch.cat([grid_tensor_cpu, padding], dim=1)
                
                log_prob = target_cpu.log_prob(grid_tensor_cpu).numpy()
                target_log_prob_grid = log_prob.reshape(grid_points, grid_points)
            
            target_cpu = target_cpu.to(target_device)
        except Exception as e:
            print(f"  Warning: Could not compute target density: {e}")
    
    # Plot each list of timesteps
    if len(timesteps) > 1:
        for group_idx, ts_list in enumerate(timesteps):
            _plot_velocity_group(ts_list, wrapper, grid_points_tensor, xx, yy,
                                grid_points, grid_range, target_log_prob_grid, model_type,
                                args, device, save_dir, step, filename_suffix=f"_group{group_idx+1}")
        
        # Also plot all combined
        all_timesteps = [t for ts_list in timesteps for t in ts_list]
        _plot_velocity_group(all_timesteps, wrapper, grid_points_tensor, xx, yy,
                            grid_points, grid_range, target_log_prob_grid, model_type,
                            args, device, save_dir, step, filename_suffix="_all")
    else:
        # Single group
        _plot_velocity_group(timesteps[0], wrapper, grid_points_tensor, xx, yy,
                            grid_points, grid_range, target_log_prob_grid, model_type,
                            args, device, save_dir, step)


def _plot_velocity_group(ts_list, wrapper, grid_points_tensor, xx, yy,
                        grid_points, grid_range, target_log_prob_grid, model_type,
                        args, device, save_dir, step, filename_suffix=""):
    """
    Plot velocity fields: 3x3 grid and 1x6 line layout.
    """
    # ============ 3x3 GRID PLOT ============
    n_grid_times = min(9, len(ts_list))
    grid_step = max(1, len(ts_list) // n_grid_times)
    ts_grid = ts_list[::grid_step][:n_grid_times]
    
    n_cols = 3
    n_rows = (n_grid_times + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, t in enumerate(ts_grid):
        with torch.no_grad():
            if model_type in ["learned", "linear"]:
                v = wrapper(torch.tensor(t, dtype=torch.float32, device=device), grid_points_tensor)
            else:
                v = wrapper(torch.tensor(t, dtype=torch.float32, device=device), grid_points_tensor)
            v = v.cpu().numpy()
        
        vx = v[:, 0].reshape(grid_points, grid_points)
        vy = v[:, 1].reshape(grid_points, grid_points)
        magnitude = np.sqrt(vx**2 + vy**2)
        magnitude_safe = np.where(magnitude > 1e-10, magnitude, 1.0)
        vx_normalized = vx / magnitude_safe
        vy_normalized = vy / magnitude_safe
        
        ax = axes[idx]
        
        if target_log_prob_grid is not None:
            log_prob_min = float(np.quantile(target_log_prob_grid[~np.isinf(target_log_prob_grid)], 0.1))
            log_prob_clipped = np.clip(target_log_prob_grid, log_prob_min, None)
            ax.contour(xx, yy, log_prob_clipped, levels=10, colors='black', alpha=1, linewidths=1)
        
        quiv = ax.quiver(xx, yy, vx_normalized, vy_normalized, magnitude, 
                        cmap='viridis', scale=30, width=0.005, headwidth=3, headlength=4)
        
        cbar = plt.colorbar(quiv, ax=ax, label=r'$\|v_t(x)\|_2$')
        
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[0], grid_range[1])
        ax.set_title(r'$v_t \text{ for } t=' + f'{t:.2f}' + '$')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    for idx in range(n_grid_times, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    step_str = f"step_{step}" if step is not None else "final"
    if model_type in ["learned", "linear"]:
        sigma = getattr(args, 'sigma', 1.0)
        fname = f"velocity_field_{model_type}_sig{sigma}_{step_str}{filename_suffix}_grid.png"
    else:
        fname = f"velocity_field_{model_type}_{step_str}{filename_suffix}_grid.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved velocity field 3x3 grid plot: {fname}")
    
    # ============ 1x6 LINE PLOT ============
    n_line_times = min(6, len(ts_list))
    line_step = max(1, len(ts_list) // n_line_times)
    ts_line = ts_list[::line_step][:n_line_times]
    
    fig, axes = plt.subplots(1, n_line_times, figsize=(n_line_times * 2.2, 2.2))
    if n_line_times == 1:
        axes = [axes]
    
    for idx, t in enumerate(ts_line):
        with torch.no_grad():
            if model_type in ["learned", "linear"]:
                v = wrapper(torch.tensor(t, dtype=torch.float32, device=device), grid_points_tensor)
            else:
                v = wrapper(torch.tensor(t, dtype=torch.float32, device=device), grid_points_tensor)
            v = v.cpu().numpy()
        
        vx = v[:, 0].reshape(grid_points, grid_points)
        vy = v[:, 1].reshape(grid_points, grid_points)
        magnitude = np.sqrt(vx**2 + vy**2)
        magnitude_safe = np.where(magnitude > 1e-10, magnitude, 1.0)
        vx_normalized = vx / magnitude_safe
        vy_normalized = vy / magnitude_safe
        
        ax = axes[idx]
        
        if target_log_prob_grid is not None:
            log_prob_min = float(np.quantile(target_log_prob_grid[~np.isinf(target_log_prob_grid)], 0.1))
            log_prob_clipped = np.clip(target_log_prob_grid, log_prob_min, None)
            ax.contour(xx, yy, log_prob_clipped, levels=10, colors='black', alpha=1, linewidths=1)
        
        quiv = ax.quiver(xx, yy, vx_normalized, vy_normalized, magnitude, 
                        cmap='viridis', scale=30, width=0.005, headwidth=3, headlength=4)
        
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[0], grid_range[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_edgecolor('black')
        ax.spines['right'].set_edgecolor('black')
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.text(0.5, 0.95, f't = {t:.2f}', transform=ax.transAxes, 
                ha='center', va='top', fontsize=8, color='black', weight='bold')
        ax.set_aspect('equal')
    
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0, right=1, top=1, bottom=0)
    
    step_str = f"step_{step}" if step is not None else "final"
    if model_type in ["learned", "linear"]:
        sigma = getattr(args, 'sigma', 1.0)
        fname = f"velocity_field_{model_type}_sig{sigma}_{step_str}{filename_suffix}_line.png"
    else:
        fname = f"velocity_field_{model_type}_{step_str}{filename_suffix}_line.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Saved velocity field 1x6 line plot: {fname}")


def plot_trajectories(full_trajectories, trajectory_t_span, target=None, grid_range=(-15, 15), 
                      save_dir=None, model_type=None, args=None):
    """
    Plot sampled trajectories at 9 evenly spaced timesteps (3x3 grid) and 6 evenly spaced timesteps (1x6 line).
    
    Args:
        full_trajectories (list): List of trajectories, each shape (num_timesteps, dim)
        trajectory_t_span (np.ndarray): Time points corresponding to trajectories
        target: Target distribution (for plotting contours)
        grid_range (tuple): Grid bounds for plotting
        save_dir (str): Directory to save plots
        model_type (str): Model type (for filename)
        args: Configuration (for sigma)
    """
    if not full_trajectories or trajectory_t_span is None:
        print("  No trajectories to plot")
        return
    
    if save_dir is None:
        save_dir = "trajectory_plots"
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert trajectories to numpy
    trajectories_np = np.array([traj[:, :2] for traj in full_trajectories])  # (num_trajs, num_timesteps, 2)
    
    # Check if target is Funnel - if so, flip x1 and x2 so funnel "looks down"
    from fab.target_distributions.funnel import Funnel
    is_funnel = isinstance(target, Funnel)
    if is_funnel:
        # Swap columns: [x1, x2] -> [x2, x1]
        trajectories_np = trajectories_np[:, :, [1, 0]]
    
    # Precompute target density on grid for contours
    target_log_prob_grid = None
    xx, yy = None, None
    if target is not None:
        try:
            x_range = np.linspace(grid_range[0], grid_range[1], 50)
            y_range = np.linspace(grid_range[0], grid_range[1], 50)
            xx, yy = np.meshgrid(x_range, y_range)
            grid_points_xy = np.stack([xx.flatten(), yy.flatten()], axis=1)
            
            # If Funnel, flip grid points to match flipped trajectories
            if is_funnel:
                grid_points_xy = grid_points_xy[:, [1, 0]]  # Swap columns
            
            target_device = target.device if hasattr(target, 'device') else 'cpu'
            target_cpu = target.to('cpu')
            
            with torch.no_grad():
                grid_tensor_cpu = torch.from_numpy(grid_points_xy).float()
                if grid_tensor_cpu.shape[1] < target_cpu.dim:
                    padding = torch.zeros(grid_tensor_cpu.shape[0], target_cpu.dim - 2)
                    grid_tensor_cpu = torch.cat([grid_tensor_cpu, padding], dim=1)
                
                log_prob = target_cpu.log_prob(grid_tensor_cpu).numpy()
                target_log_prob_grid = log_prob.reshape(50, 50)
            
            target_cpu = target_cpu.to(target_device)
        except Exception as e:
            print(f"  Warning: Could not compute target density: {e}")
    
    # ============ 3x3 GRID PLOT ============
    num_timesteps = len(trajectory_t_span)
    timestep_indices_grid = np.linspace(0, num_timesteps - 1, 9, dtype=int)
    selected_times_grid = trajectory_t_span[timestep_indices_grid]
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    for idx, (ax, t_idx, t_val) in enumerate(zip(axes, timestep_indices_grid, selected_times_grid)):
        positions_at_t = trajectories_np[:, t_idx, :]
        
        # Plot trajectories using flipped version
        for traj_idx in range(len(full_trajectories)):
            traj_2d = trajectories_np[traj_idx, :t_idx+1, :]
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.15, color='gray', linewidth=0.5)
        
        if target_log_prob_grid is not None and xx is not None:
            log_prob_min = float(np.quantile(target_log_prob_grid[~np.isinf(target_log_prob_grid)], 0.1))
            log_prob_clipped = np.clip(target_log_prob_grid, log_prob_min, None)
            ax.contour(xx, yy, log_prob_clipped, levels=8, colors='black', alpha=0.3, linewidths=0.8)
        
        ax.scatter(positions_at_t[:, 0], positions_at_t[:, 1], alpha=0.6, s=3, c='darkblue')
        
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[0], grid_range[1])
        ax.set_title(f't = {t_val:.3f}')
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$x_2$')
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if model_type in ["learned", "linear"]:
        sigma = getattr(args, 'sigma', 1.0)
        fname = f"trajectories_{model_type}_sig{sigma}_grid_final.png"
    else:
        fname = f"trajectories_{model_type}_grid_final.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved trajectory 3x3 grid plot: {fname}")
    
    # ============ 1x6 LINE PLOT ============
    n_line_times = 6
    line_times = np.linspace(trajectory_t_span[0], trajectory_t_span[-1], n_line_times)
    line_indices = [np.argmin(np.abs(trajectory_t_span - t)) for t in line_times]
    line_times_actual = trajectory_t_span[line_indices]
    
    fig, axes = plt.subplots(1, n_line_times, figsize=(n_line_times * 2.2, 2.2))
    if n_line_times == 1:
        axes = [axes]
    
    for idx, (ax, t_idx, t_val) in enumerate(zip(axes, line_indices, line_times_actual)):
        positions_at_t = trajectories_np[:, t_idx, :]
        
        # Plot trajectories using flipped version
        for traj_idx in range(len(full_trajectories)):
            traj_2d = trajectories_np[traj_idx, :t_idx+1, :]
            ax.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.15, color='gray', linewidth=0.5)
        
        if target_log_prob_grid is not None and xx is not None:
            log_prob_min = float(np.quantile(target_log_prob_grid[~np.isinf(target_log_prob_grid)], 0.1))
            log_prob_clipped = np.clip(target_log_prob_grid, log_prob_min, None)
            ax.contour(xx, yy, log_prob_clipped, levels=8, colors='black', alpha=0.3, linewidths=0.8)
        
        ax.scatter(positions_at_t[:, 0], positions_at_t[:, 1], alpha=0.6, s=3, c='darkblue')
        
        ax.set_xlim(grid_range[0], grid_range[1])
        ax.set_ylim(grid_range[0], grid_range[1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['top'].set_edgecolor('black')
        ax.spines['right'].set_edgecolor('black')
        ax.spines['bottom'].set_edgecolor('black')
        ax.spines['left'].set_edgecolor('black')
        ax.text(0.5, 0.95, f't = {t_val:.2f}', transform=ax.transAxes, 
                ha='center', va='top', fontsize=9, color='black', weight='bold')
        ax.set_aspect('equal')
    
    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0, right=1, top=1, bottom=0)
    
    if model_type in ["learned", "linear"]:
        sigma = getattr(args, 'sigma', 1.0)
        fname = f"trajectories_{model_type}_sig{sigma}_line_final.png"
    else:
        fname = f"trajectories_{model_type}_line_final.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"  Saved trajectory 1x6 line plot: {fname}")


def get_eval_callbacks(target_name):
    """
    Get plotting and metrics callbacks for a specific target.
    
    Args:
        target_name (str): Name of the target (e.g., "Funnel", "AsymmetricGMM").
        
    Returns:
        tuple: (plotting_callback, metrics_callback)
    """
    if target_name == "Funnel":
        def funnel_plotting(x_gen, eps_kept, step, save_dir, big_eval, target, models=None, model_type=None, args=None, device=None, full_trajectories=None, trajectory_t_span=None):
            
            # Create sampler wrapper for plotting compatibility
            class SamplerWrapper:
                def __init__(self, funnel_dist):
                    self.funnel = funnel_dist
                    self.scale1 = funnel_dist.sigma
                def __getattr__(self, name):
                    return getattr(self.funnel, name)   
                def log_prob(self, x):
                    if hasattr(self.funnel, '_sigma2'):
                        target_device = self.funnel._sigma2.device
                        x = x.to(target_device)
                    return self.funnel.log_prob(x)
                def sample(self, n, device=None, dtype=None):
                    samples = self.funnel.sample((n,))
                    if device is None: samples = samples.cpu()
                    elif device is not None: samples = samples.to(device)
                    if dtype is not None: samples = samples.to(dtype)
                    return samples
            
            sampler = SamplerWrapper(target)
            
            # Extract first 2 dimensions
            x_gen_2d = x_gen[:, :2].cpu()
            eps_2d = eps_kept[:, :2] if eps_kept.shape[1] >= 2 else eps_kept
            
            # Construct filenames
            if step is not None:
                fname = f"samples_step_{step}.png"
                latent_fname = f"latent_color_step_{step}.png"
            else:
                fname = "samples_final.png"
                latent_fname = "latent_color_final.png"

            # Plot
            # Pass step=0 as dummy because we provide filename
            plot_funnel_2d(x_gen_2d, sampler, step=0, big_eval=big_eval, path=save_dir, filename=fname)
            plot_latent_colored_by_target_norm(eps_2d, x_gen_2d, step=0, path=save_dir, big_eval=big_eval, filename=latent_fname)
            
            # Only plot velocity field and trajectories for final evaluation
            if step is None:
                # Plot velocity field if enabled
                if getattr(args, 'plot_velocity', True):
                    try:
                        from fab.utils.funnel_plotting import funnel_axis_limits
                        
                        scale1 = float(getattr(target, "sigma", 3.0))
                        divide_by_two = getattr(target, "divide_variance_by_two", True)
                        dim = getattr(target, "dim", 2)
                        
                        # Use same bounds as funnel_2d plotting
                        x1_min, x1_max = max(-10, -10*1/np.sqrt(scale1)), 10 * np.sqrt(scale1)
                        x2_min, x2_max = funnel_axis_limits(scale1, dim, divide_by_two)
                        
                        grid_range = (min(x1_min, x2_min), max(x1_max, x2_max))
                        grid_size = 35
                        plot_velocity_field(models, model_type, args, device, target=target, 
                                          grid_range=grid_range, grid_points=grid_size, save_dir=save_dir, step=step)
                    except Exception as e:
                        print(f"  Velocity field plotting failed: {e}")
                        traceback.print_exc()
                
                # Plot trajectories if enabled
                if getattr(args, 'plot_trajectories', True) and full_trajectories is not None:
                    try:
                        from fab.utils.funnel_plotting import funnel_axis_limits
                        
                        scale1 = float(getattr(target, "sigma", 3.0))
                        divide_by_two = getattr(target, "divide_variance_by_two", True)
                        dim = getattr(target, "dim", 2)
                        
                        # Use same bounds as funnel_2d plotting
                        x1_min, x1_max = max(-10, -10*1/np.sqrt(scale1)), 10 * np.sqrt(scale1)
                        x2_min, x2_max = funnel_axis_limits(scale1, dim, divide_by_two)
                        
                        grid_range = (min(x1_min, x2_min), max(x1_max, x2_max))
                        plot_trajectories(full_trajectories, trajectory_t_span, target=target, 
                                        grid_range=grid_range, save_dir=save_dir, model_type=model_type, args=args)
                    except Exception as e:
                        print(f"  Trajectory plotting failed: {e}")
                        traceback.print_exc()

        def funnel_metrics(x_gen, eps_kept, step, save_dir, big_eval, target):            
            x_gen_2d = x_gen[:, :2].cpu()
            scale1 = float(getattr(target, "sigma", 3.0))
            
            metrics = funnel_eval.evaluate_x2_marginal_metrics(
                x_gen_2d, scale1=scale1, tail_q=1e-3, gh_n=200
            )
            return metrics
            
        return funnel_plotting, funnel_metrics

    elif target_name == "AsymmetricGMM" or target_name == "GMM":
        def gmm_plotting(x_gen, eps_kept, step, save_dir, big_eval, target, models=None, model_type=None, args=None, device=None, full_trajectories=None, trajectory_t_span=None):
            # Extract 2D samples for plotting
            x_gen_2d = x_gen[:, :2].cpu()
            
            # Cap both at same number
            num_plot_samples = min(5000, x_gen_2d.shape[0])
            x_gen_plot = x_gen_2d[:num_plot_samples]
            
            # draw true samples for comparison
            try:
                true_samples = target.sample((num_plot_samples,))[:, :2].cpu()
            except Exception:
                true_samples = target.sample(num_plot_samples)[:, :2].cpu()

            # Fixed plotting limit
            plot_min = -target.mean_offset - 8  
            plot_max = 4 + 8                        
            bounds = (plot_min, plot_max)

            # Move target to CPU for plotting
            target_device = target.device if hasattr(target, 'device') else 'cpu'
            target = target.to('cpu')

            # Create a wrapper for the 2D log_prob (need to pad back to full dim)
            def log_prob_2d(x_2d):
                if x_2d.shape[1] < target.dim:
                    padding = torch.zeros(x_2d.shape[0], target.dim - 2, device=x_2d.device)
                    x_full = torch.cat([x_2d, padding], dim=1)
                else:
                    x_full = x_2d
                return target.log_prob(x_full)

            # Create scatter plot (two panels)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

            # Plot generated samples with target contours
            ax1.scatter(x_gen_plot[:, 0], x_gen_plot[:, 1], alpha=0.5, s=2, c='darkblue', label='Generated')
            plot_contours_mixture(log_prob_2d, ax=ax1, bounds=bounds, grid_width_n_points=100, n_contour_levels=15)
            ax1.set_xlabel(r'$x_1$')
            ax1.set_ylabel(r'$x_2$')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_xlim(bounds[0], bounds[1])
            ax1.set_ylim(bounds[0], bounds[1])

            # Plot true samples with target contours
            ax2.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, s=2, c='darkred', label='True')
            plot_contours_mixture(log_prob_2d, ax=ax2, bounds=bounds, grid_width_n_points=100, n_contour_levels=15)
            ax2.set_title('True Samples', fontsize=12)
            ax2.set_xlabel(r'$x_1$')
            ax2.set_ylabel(r'$x_2$')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_xlim(bounds[0], bounds[1])
            ax2.set_ylim(bounds[0], bounds[1])

            plt.tight_layout()

            checkpoint_label = os.path.basename(os.path.dirname(save_dir))
            combined_fname = os.path.join(save_dir, f"{checkpoint_label}.png")
            fig.savefig(combined_fname, dpi=150, bbox_inches='tight')
            plt.close(fig)

            fig2, ax = plt.subplots(1, 1, figsize=(9, 8), dpi=150)
            ax.scatter(x_gen_2d[:, 0], x_gen_2d[:, 1], alpha=0.8, s=2, c='darkblue')
            plot_contours_mixture(log_prob_2d, ax=ax, bounds=bounds, grid_width_n_points=100, n_contour_levels=8)
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(bounds[0], bounds[1])
            ax.set_ylim(bounds[0], bounds[1])
            gen_fname = os.path.join(save_dir, f"{checkpoint_label}_generated.png")
            fig2.tight_layout()
            fig2.savefig(gen_fname, dpi=150, bbox_inches='tight')
            plt.close(fig2)

            # Move target back to original device
            target = target.to(target_device)

            print(f"  Saved plots to {combined_fname}")
            
            
            fig4, ax = plt.subplots(1, 1, figsize=(9, 8), dpi=150)  
            ax.scatter(x_gen_2d[:, 0], x_gen_2d[:, 1], alpha=0.8, s=2, c='darkblue', label='Generated')
            ax.set_xlabel(r'$x_1$')
            ax.set_ylabel(r'$x_2$')
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig4.tight_layout()
            nocontour_fname = os.path.join(save_dir, f"{checkpoint_label}_nocontour.png")
            fig4.savefig(nocontour_fname, dpi=150, bbox_inches='tight')
            plt.close(fig4)
            print(f"  Saved no-contour plot to {nocontour_fname}")
            
            # Also save a simple scatter plot without contours or box limits
            fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
            ax1.scatter(x_gen_2d[:, 0], x_gen_2d[:, 1], alpha=0.5, s=2, c='darkblue', label='Generated')
            ax1.set_xlabel(r'$x_1$')
            ax1.set_ylabel(r'$x_2$')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.5, s=2, c='darkred', label='True')
            ax2.set_title('True Samples', fontsize=12)
            ax2.set_xlabel(r'$x_1$')
            ax2.set_ylabel(r'$x_2$')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            plt.tight_layout()
            simple_fname = os.path.join(save_dir, f"{checkpoint_label}_simple.png")
            fig3.savefig(simple_fname, dpi=150, bbox_inches='tight')
            plt.close(fig3)
            

            print(f"  Saved simple plot to {simple_fname}")
            
            # Only plot velocity field and trajectories for final evaluation
            if step is None:
                # Plot velocity field if plot_velocity is enabled
                if getattr(args, 'plot_velocity', True) and models is not None and model_type is not None:
                    try:
                        plot_min = -target.mean_offset - 8
                        plot_max = 4 + 8
                        grid_range = (plot_min, plot_max)
                        grid_size = max(15, int((plot_max - plot_min) / 1.5))
                        plot_velocity_field(models, model_type, args, device, target=target, 
                                          grid_range=grid_range, grid_points=grid_size, save_dir=save_dir, step=step)
                    except Exception as e:
                        print(f"  Velocity field plotting failed: {e}")
                        traceback.print_exc()
                
                # Plot trajectories if enabled
                if getattr(args, 'plot_trajectories', True) and full_trajectories is not None:
                    try:
                        plot_min = -target.mean_offset - 8
                        plot_max = 4 + 8
                        grid_range = (plot_min, plot_max)
                        plot_trajectories(full_trajectories, trajectory_t_span, target=target, 
                                        grid_range=grid_range, save_dir=save_dir, model_type=model_type, args=args)
                    except Exception as e:
                        print(f"  Trajectory plotting failed: {e}")
                        traceback.print_exc()
                
        def gmm_metrics(x_gen, eps_kept, step, save_dir, big_eval, target):
            # Energy distance and ESS computed in run_heavy_evaluation
            return {}
            
        return gmm_plotting, gmm_metrics
    
    elif target_name == "Rings":
        def rings_plotting(x_gen, eps_kept, step, save_dir, big_eval, target, models=None, model_type=None, args=None, device=None, full_trajectories=None, trajectory_t_span=None):
            
            # Cap both at same number for fair comparison
            num_plot_samples = min(5000, x_gen.shape[0])
            x_gen_plot = x_gen[:num_plot_samples]
            
            # Draw true samples for comparison
            try:
                true_samples = target.sample((num_plot_samples,)).cpu()
            except Exception:
                true_samples = target.sample(num_plot_samples).cpu()
            
            # Set plotting bounds based on rings configuration
            plot_range = target.num_modes * target.radius + 3 * target.sigma
            bounds = (-plot_range, plot_range)
            
            # Move target to CPU for plotting
            target_device = target.device if hasattr(target, 'device') else 'cpu'
            target = target.to('cpu')
            
            # Create 2D log_prob wrapper (rings are already 2D)
            def log_prob_2d(x_2d):
                return target.log_prob(x_2d)
            
            # Construct filenames
            if step is not None:
                fname_samples = f"samples_step_{step}.png"
                fname_comparison = f"comparison_step_{step}.png"
            else:
                fname_samples = "samples_final.png"
                fname_comparison = "comparison_final.png"
            
            # Plot generated samples
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.scatter(x_gen_plot[:, 0], x_gen_plot[:, 1], alpha=0.3, s=1, label='Generated')
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_aspect('equal')
            ax.legend()
            plt.savefig(os.path.join(save_dir, fname_samples), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Plot comparison with true samples
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Generated samples
            axes[0].scatter(x_gen_plot[:, 0], x_gen_plot[:, 1], alpha=0.3, s=1, c='blue')
            axes[0].set_xlim(bounds)
            axes[0].set_ylim(bounds)
            axes[0].set_title('Generated Samples')
            axes[0].set_xlabel('$x_1$')
            axes[0].set_ylabel('$x_2$')
            axes[0].set_aspect('equal')
            
            # True samples
            axes[1].scatter(true_samples[:, 0], true_samples[:, 1], alpha=0.3, s=1, c='red')
            axes[1].set_xlim(bounds)
            axes[1].set_ylim(bounds)
            axes[1].set_title('True Samples')
            axes[1].set_xlabel('$x_1$')
            axes[1].set_ylabel('$x_2$')
            axes[1].set_aspect('equal')
            
            plt.savefig(os.path.join(save_dir, fname_comparison), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Restore target device
            target.to(target_device)
            
            # Only plot velocity field and trajectories for final evaluation
            if step is None:
                # Plot velocity field if enabled
                if getattr(args, 'plot_velocity', True) and models is not None:
                    try:
                        grid_range = bounds
                        grid_size = 35
                        plot_velocity_field(models, model_type, args, device, target=target, 
                                          grid_range=grid_range, grid_points=grid_size, save_dir=save_dir, step=step)
                    except Exception as e:
                        print(f"  Velocity field plotting failed: {e}")
                        traceback.print_exc()
                
                # Plot trajectories if enabled
                if getattr(args, 'plot_trajectories', True) and full_trajectories is not None:
                    try:
                        grid_range = bounds
                        plot_trajectories(full_trajectories, trajectory_t_span, target=target, 
                                        grid_range=grid_range, save_dir=save_dir, model_type=model_type, args=args)
                    except Exception as e:
                        print(f"  Trajectory plotting failed: {e}")
                        traceback.print_exc()
        
        def rings_metrics(x_gen, eps_kept, step, save_dir, big_eval, target):
            # Energy distance and ESS computed in run_heavy_evaluation
            return {}
        
        return rings_plotting, rings_metrics
        
    else:
        # Default generic callbacks
        def generic_plotting(x_gen, eps_kept, step, save_dir, big_eval, target, models=None, model_type=None, args=None, device=None, full_trajectories=None, trajectory_t_span=None):
            if x_gen.shape[1] >= 2:
                x_gen = x_gen.cpu().numpy()
                plt.figure(figsize=(8, 8))
                plt.scatter(x_gen[:, 0], x_gen[:, 1], alpha=0.5, s=1)
                step_str = str(step) if step is not None else "final"
                plt.title(f"Samples at step {step_str}")
                
                if step is not None:
                    fname = f"samples_{step}.png"
                else:
                    fname = "samples_final.png"
                    
                plt.savefig(os.path.join(save_dir, fname))
                plt.close()
                
        return generic_plotting, None