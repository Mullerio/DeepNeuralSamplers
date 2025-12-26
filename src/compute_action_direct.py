"""
Direct action computation without subprocess calls.
Works directly with models, target, and config objects.
"""

import torch
import numpy as np
from torchdiffeq import odeint
from tqdm import tqdm
import os
from loss import beta, beta_int
from eval_utils import cnf_sample
from loss import TorchWrapper, TorchWrapperGF
import matplotlib.pyplot as plt
import matplotlib as mpl

# matplotlib mathtex 
mpl.rcParams['text.usetex'] = False
mpl.rcParams['mathtext.fontset'] = 'cm'  
mpl.rcParams['font.family'] = 'serif'

def compute_action_direct(models, target, args, device,
                        model_type="linear",
                        num_samples=2000,
                        num_time_steps=100,
                        batch_size=200,
                        output_dir=None,
                        skip_nan=True):
    """
    Compute action over time
    """

    print(f"\nComputing action integral (OLD LOGIC)")
    print(f" Model type: {model_type}")
    print(f" Number of samples: {num_samples}")
    print(f" Time discretization: {num_time_steps} points")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Time discretization
    time_points = torch.linspace(0, 1, num_time_steps).to(device)
    dt = time_points[1] - time_points[0]
    dim = args.dim

    # Velocity wrapper
    if model_type in ["learned", "linear"]:
        if hasattr(models, "velo"):
            velo = models.velo
        else:
            velo = models["velo"]
        wrapper = TorchWrapper(velo)

    elif model_type == "gf":
        if hasattr(models, "psi"):
            psi = models.psi
            schedule = models.schedule
        else:
            psi = models["psi"]
            schedule = models["schedule"]

        beta_min = getattr(args, 'beta_min', 0.1)
        beta_max = getattr(args, 'beta_max', 20.0)
        beta_fn = lambda t: beta(t, beta_min, beta_max)

        wrapper = TorchWrapperGF(psi, schedule, target, beta_fn)

    else:
        raise ValueError(f"Unknown model_type {model_type}")

    action_per_time = np.zeros(num_time_steps)
    nan_count_per_time = np.zeros(num_time_steps, dtype=int)
    nan_time_indices = []

    num_batches = (num_samples + batch_size - 1) // batch_size
    total_samples_processed = 0
    num_nan_batches = 0

    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        current_batch_size = min(batch_size, num_samples - total_samples_processed)
        if current_batch_size <= 0:
            break

        with torch.no_grad():
            x0 = torch.randn(current_batch_size, dim, device=device)

            trajectory = [x0]
            current_x = x0

            has_nan = False
            for i in range(1, num_time_steps):
                t_prev = time_points[i-1]

                velocity = wrapper(t_prev.unsqueeze(0).expand(current_batch_size, 1), current_x)
                current_x = current_x + velocity * dt

                if torch.isnan(current_x).any() or torch.isnan(velocity).any():
                    has_nan = True
                    nan_mask = torch.isnan(current_x).any(dim=1) | torch.isnan(velocity).any(dim=1)
                    nan_count_per_time[i] += nan_mask.sum().item()
                    if i not in nan_time_indices:
                        nan_time_indices.append(i)

                    if skip_nan:
                        num_nan_batches += 1
                        break  
                    else:
                        raise ValueError(f"NaN detected in trajectory at time step {i}.")

                trajectory.append(current_x)

            if has_nan and skip_nan:
                continue

            for i, t in enumerate(time_points):
                x_t = trajectory[i]
                v_t = wrapper(t.unsqueeze(0).expand(current_batch_size, 1), x_t)

                if torch.isnan(v_t).any():
                    nan_mask = torch.isnan(v_t).any(dim=1)
                    nan_count_per_time[i] += nan_mask.sum().item()
                    if i not in nan_time_indices:
                        nan_time_indices.append(i)

                    if skip_nan:
                        num_nan_batches += 1
                        continue  
                    else:
                        raise ValueError(f"NaN detected in velocity at time step {i}.")

                v_squared = (v_t ** 2).sum(dim=1)  # Shape: (batch_size,)

                # Filter out NaN values if flag is set
                if skip_nan:
                    valid_mask = ~torch.isnan(v_squared)
                    if valid_mask.any():
                        action_per_time[i] += v_squared[valid_mask].sum().item()
                    num_nan_in_batch = (~valid_mask).sum().item()
                    if num_nan_in_batch > 0:
                        nan_count_per_time[i] += num_nan_in_batch
                        if i not in nan_time_indices:
                            nan_time_indices.append(i)
                else:
                    action_per_time[i] += v_squared.sum().item()

        total_samples_processed += current_batch_size

    total_nan_datapoints = nan_count_per_time.sum()
    if total_nan_datapoints > 0:
        print(f"\n{'='*60}")
        print(f"NaN Statistics:")
        print(f"  Total NaN datapoints: {total_nan_datapoints}")
        print(f"  NaN datapoints as % of total: {100 * total_nan_datapoints / (num_samples * num_time_steps):.2f}%")
        #print(f"  Time indices with NaN: {sorted(nan_time_indices)}")
        #print(f"  Number of time steps affected: {len(nan_time_indices)}")
        print(f"{'='*60}\n")

    # Normalize by number of samples
    action_per_time /= max(total_samples_processed, 1)

    # Integrate over time using trapezoidal rule
    total_action = np.trapz(action_per_time, dx=dt.item())

    print(f"\nResults:")
    print(f"  Total action: {total_action:.6f}")
    print(f"  Mean action per time: {action_per_time.mean():.6f}")
    print(f"  Max action per time: {action_per_time.max():.6f}")
    print(f"  Min action per time: {action_per_time.min():.6f}")

    # Save results
    results = {
        'total_action': total_action,
        'action_per_time': action_per_time,
        'time_points': time_points.cpu().numpy(),
        'num_samples': total_samples_processed,
        'model_type': model_type,
        'nan_count_per_time': nan_count_per_time,
        'nan_time_indices': np.array(sorted(nan_time_indices)),
        'total_nan_datapoints': total_nan_datapoints,
    }

    if output_dir:
        np.savez(os.path.join(output_dir, 'action_results_old_logic.npz'), **results)
        plot_action(results, output_dir)
        print(f"\nSaved results to {output_dir}/action_results_old_logic.npz")

    return results





def plot_action(results, output_dir):

    time_points = results['time_points']
    action_per_time = results['action_per_time']
    total_action = results['total_action']

    dt = time_points[1] - time_points[0]
    cumulative_action = np.cumsum(action_per_time) * dt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(time_points, action_per_time, linewidth=2)
    axes[0].set_xlabel(r"$t$")
    axes[0].set_ylabel(r"$\int \: \:  \|\mathbf{v}_t\|^2 d\mu_t$")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_points, cumulative_action, linewidth=2, color='orange')
    axes[1].set_xlabel(r"$t$")
    axes[1].set_ylabel("Cumulative Action")
    axes[1].set_title(f"Cumulative Action: {total_action:.4f}")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_plot.png"), dpi=150)
    plt.close()


    plt.figure(figsize=(6,4))
    plt.plot(time_points, action_per_time, linewidth=2)
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\int \: \: \|\mathbf{v}_t\|^2 d\mu_t$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_energy.png"), dpi=150)
    plt.close()


    plt.figure(figsize=(6,4))
    plt.plot(time_points, cumulative_action, linewidth=2, color='orange')
    plt.xlabel(r"$t$")
    plt.ylabel("Cumulative Action")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "action_cumulative.png"), dpi=150)
    plt.close()

    print("  Saved: action_plot.png, action_energy.png, action_cumulative.png")
