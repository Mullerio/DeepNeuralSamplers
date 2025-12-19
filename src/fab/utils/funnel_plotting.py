import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
from scipy.special import gamma
from fab.target_distributions.funnel import Funnel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FuncFormatter

# =====================================================
# NEW: Unified funnel axis scaling
# =====================================================
def funnel_axis_limits(sigma, dim, divide_variance_by_two: bool):
    """
    Compute good visual plot bounds for r = ||x_{2:d}|| or x2 dimension.
    
    Based on:
        r_max ≈ k * sqrt(exp(3σ)/c) * sqrt(dim - 1)
    
    c = 1  if variance = exp(x1)
    c = 2  if variance = exp(x1)/2
    
    k is a visual widening factor:
        - Divide-case looks best with k=2.5
        - Non-divide looks dramatic with k=8.0
    """
    c = 2 if divide_variance_by_two else 2
    k = 8 if divide_variance_by_two else 13.0

    r_max = k * np.sqrt(np.exp(3 * sigma) / c) * np.sqrt(max(1, dim - 1))
    return -r_max, r_max


# =====================================================
# Compute joint log-prob grid
# =====================================================
def compute_log_joint_grid(
        sampler, 
        x1_range, 
        x2_range,
        device="cuda", 
        n1=300, 
        n2=300
    ):
    x1_lin = torch.linspace(*x1_range, n1, device=device)
    x2_lin = torch.linspace(*x2_range, n2, device=device)

    X1, X2 = torch.meshgrid(x1_lin, x2_lin, indexing="ij")
    
    grid_torch = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=-1)

    target_dim = getattr(sampler, 'dim', 2)

    if target_dim > 2:
        from fab.target_distributions.funnel import Funnel
        
        sigma_val = getattr(sampler, 'sigma', 3.0)
        if hasattr(sigma_val, 'item'):
            sigma_val = float(sigma_val.item())
        
        funnel_2d = Funnel(dim=2, sigma=sigma_val, use_gpu=(device == 'cuda'))
        logp = funnel_2d.log_prob(grid_torch).detach().cpu().numpy().reshape(n1, n2)
    else:
        logp = sampler.log_prob(grid_torch).detach().cpu().numpy().reshape(n1, n2)

    return x1_lin.cpu().numpy(), x2_lin.cpu().numpy(), logp


# =====================================================
# Analytic funnel marginal p(x2)
# =====================================================
def _analytic_funnel_x2_pdf(x2_grid: np.ndarray, scale1: float, divide_by_two: bool = False, gh_n: int = 80) -> np.ndarray:
    from numpy.polynomial.hermite import hermgauss

    x2 = x2_grid.astype(np.float64)
    nodes, weights = hermgauss(gh_n)
    x1_vals = (np.sqrt(2.0) * scale1) * nodes
    w_norm = weights / np.sqrt(np.pi)

    var = np.exp(x1_vals)
    if divide_by_two:
        var = var / 2.0
    inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

    var_col = var[:, None]
    coef = inv_sqrt_2pi / np.sqrt(var_col)
    expo = np.exp(- (x2[None, :]**2) / (2.0 * var_col))
    pdf_matrix = coef * expo
    pdf = (w_norm[:, None] * pdf_matrix).sum(axis=0)

    return np.maximum(pdf, 1e-300)


# =====================================================
# Helper converting samples to (x1, r)
# =====================================================
def funnel_to_2d(x: torch.Tensor):
    x1 = x[:, 0:1]
    r  = torch.norm(x[:, 1:], dim=1, keepdim=True)
    return torch.cat([x1, r], dim=1)


# =====================================================
# plotting helpers
# =====================================================
@torch.no_grad()
def plot_funnel_2d(
        generated, 
        sampler, 
        step, 
        big_eval=False,
        path=None,
        filename=None
    ):

    os.makedirs(path or ".", exist_ok=True)

    # Extract sigma and whether we divide variance by 2
    scale1 = float(
        getattr(sampler, 'scale1', getattr(sampler, 'sigma', torch.tensor(3.0))).item()
        if hasattr(getattr(sampler, 'scale1', getattr(sampler, 'sigma', torch.tensor(3.0))), 'item')
        else getattr(sampler, 'scale1', getattr(sampler, 'sigma', 3.0))
    )
    divide_by_two = getattr(sampler, 'divide_variance_by_two', True)

    # x1 range
    X1_MIN, X1_MAX = min(-10, -10* np.sqrt(scale1)), 10 * np.sqrt(scale1)

    # new x2 range
    X2_MIN, X2_MAX = funnel_axis_limits(scale1, sampler.dim, divide_by_two)

    n_data = generated.shape[0]
    n_true = n_data
    S_data = sampler.sample(n_true)

    # Convert to CPU and numpy for matplotlib
    if torch.is_tensor(generated):
        generated = generated.cpu()
    if torch.is_tensor(S_data):
        S_data = S_data.cpu()

    x1_d = generated[:, 0]
    x2_d = generated[:, 1]

    x1_m = S_data[:, 0]
    x2_m = S_data[:, 1]

    # Evaluate true log density grid
    x1_lin, x2_lin, logp = compute_log_joint_grid(
        sampler,
        (X1_MIN, X1_MAX),
        (X2_MIN, X2_MAX),
        n1=320,
        n2=360,
        device=generated.device
    )

    bins_x2 = np.linspace(X2_MIN, X2_MAX, 50)
    bins_x1 = np.linspace(X1_MIN, X1_MAX, 50)

    fig = plt.figure(figsize=(12, 10), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP, 
                  left=0.15, right=0.85, bottom=0.1, top=0.95)

    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal = "#7fb8c8"
    red  = "#e74c3c"

    ax_main.set_facecolor("black")
    fig.patch.set_facecolor("white")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")

    log_floor = -20.0
    vmax = float(np.max(logp))

    im = ax_main.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    # colorbar converting log → probability
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    divider = make_axes_locatable(ax_main)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)

    def log_to_prob(x, pos):
        return f'{np.exp(x):.1e}'

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_prob))
    cbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    cbar.ax.yaxis.set_ticks_position('left')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white', labelsize=9)

    # scatter data
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.5,
                    color=teal, linewidths=0, edgecolors="none")

    ax_main.set_xlabel(r"$x_2$", color="black", fontsize=12)
    ax_main.set_ylabel(r"$x_1$", color="black", fontsize=12)
    ax_main.set_xlim(X2_MIN, X2_MAX)
    ax_main.set_ylim(X1_MIN, X1_MAX)
    ax_main.tick_params(colors='black', labelsize=10)

    # x2 marginals
    ax_top.set_yscale("log")
    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)

    x2_centers = 0.5 * (bins_x2[:-1] + bins_x2[1:])
    px2 = _analytic_funnel_x2_pdf(x2_centers, scale1=scale1, divide_by_two=divide_by_two, gh_n=80)
    ax_top.plot(x2_centers, px2, color="#1f77b4", linewidth=2.2, alpha=0.95)

    ax_top.tick_params(labelbottom=False)

    # x1 marginal
    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)

    x1_centers = 0.5 * (bins_x1[:-1] + bins_x1[1:])
    true_x1_pdf = (1.0 / (scale1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x1_centers / scale1)**2)
    ax_right.plot(true_x1_pdf, x1_centers, color="#1f77b4", linewidth=2.2, alpha=0.95)

    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    plt.tight_layout()
    if filename is not None:
        out_name = os.path.join(path, filename)
    else:
        out_name = os.path.join(path, f'samples_epoch_{step:03d}.png')
    plt.savefig(out_name, bbox_inches='tight')
    plt.close()


# =====================================================
# Generic 2D fallback (for non-funnel)
# =====================================================
def _sym_limits_from_arrays(a: np.ndarray, b: np.ndarray, q: float = 99.5,
                            x1_floor: float = 3.0, x1_ceil: float = 20.0,
                            x2_floor: float = 3.0, x2_ceil: float = 1000.0):
    both = np.concatenate([a, b], axis=0)
    ax1 = np.percentile(np.abs(both[:, 0]), q)
    ax2 = np.percentile(np.abs(both[:, 1]), q)
    r1 = float(np.clip(ax1, x1_floor, x1_ceil))
    r2 = float(np.clip(ax2, x2_floor, x2_ceil))
    return (-r1, r1), (-r2, r2)


def plot_generic_2d(generated, sampler, step, big_eval=False, path="."):
    os.makedirs(path, exist_ok=True)

    n_data = generated.shape[0]
    S_data = sampler.sample(n_data)

    x1_d, x2_d = generated[:, 0].cpu().numpy(), generated[:, 1].cpu().numpy()
    x1_m, x2_m = S_data[:, 0].cpu().numpy(), S_data[:, 1].cpu().numpy()

    gen_np = np.stack([x1_d, x2_d], axis=-1)
    data_np = np.stack([x1_m, x2_m], axis=-1)
    (x1_min, x1_max), (x2_min, x2_max) = _sym_limits_from_arrays(gen_np, data_np)

    bins_x1 = np.linspace(x1_min, x1_max, 60)
    bins_x2 = np.linspace(x2_min, x2_max, 60)

    fig = plt.figure(figsize=(8, 8), dpi=160)
    GAP = 0.05
    gs = GridSpec(4, 4, figure=fig, hspace=GAP, wspace=GAP)
    ax_main  = fig.add_subplot(gs[1:, :3])
    ax_top   = fig.add_subplot(gs[0, :3], sharex=ax_main)
    ax_right = fig.add_subplot(gs[1:, 3], sharey=ax_main)

    teal, red = "#7fb8c8", "#e74c3c"

    ax_main.scatter(x2_m, x1_m, s=4, alpha=0.25, color=red, linewidths=0)
    ax_main.scatter(x2_d, x1_d, s=6, alpha=0.6, color=teal, linewidths=0)
    ax_main.set_xlabel(r"$x_2$")
    ax_main.set_ylabel(r"$x_1$")
    ax_main.set_xlim(x2_min, x2_max)
    ax_main.set_ylim(x1_min, x1_max)

    ax_top.hist(x2_m, bins=bins_x2, density=True, histtype="step", color=red, linewidth=2.0)
    ax_top.hist(x2_d, bins=bins_x2, density=True, color=teal, alpha=0.35, edgecolor=teal)
    ax_top.tick_params(labelbottom=False)

    ax_right.hist(x1_d, bins=bins_x1, density=True, orientation="horizontal",
                  color=teal, alpha=0.35, edgecolor=teal)
    ax_right.hist(x1_m, bins=bins_x1, density=True, orientation="horizontal",
                  histtype="step", color=red, linewidth=2.0)
    ax_right.tick_params(labelleft=False)
    ax_right.set_xlabel(r"$p(x_1)$")

    out_file = os.path.join(path, f'samples_epoch_{step:03d}.pdf')
    plt.savefig(out_file, bbox_inches="tight")
    plt.close(fig)


# =====================================================
# Latent colored scatter
# =====================================================
@torch.no_grad()
def plot_latent_colored_by_target_norm(
    latent: torch.Tensor,
    targets: torch.Tensor,
    step: int,
    path: str,
    big_eval: bool = False,
    title: str = "latent colored by ||x||",
    filename: str = None
):
    os.makedirs(path or ".", exist_ok=True)

    L = latent.detach().cpu().numpy()
    X = targets.detach().cpu().numpy()
    norms = np.linalg.norm(X, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=150, constrained_layout=True)

    ax0 = axes[0]
    ax0.scatter(L[:, 0], L[:, 1], s=4, alpha=0.4, color="#808080", linewidths=0)
    ax0.set_title("drawn Gaussian (latent)")
    ax0.set_aspect('equal', 'box')
    ax0.grid(True, alpha=0.2)

    ax1 = axes[1]
    h = ax1.scatter(L[:, 0], L[:, 1], s=5, c=norms, cmap="viridis", alpha=0.7, linewidths=0)
    ax1.set_title(title)
    ax1.set_aspect('equal', 'box')
    ax1.grid(True, alpha=0.2)
    cbar = fig.colorbar(h, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("||x||")

    if filename is not None:
        out_name = filename
    else:
        out_name = f'latent_color_epoch_{int(step):03d}.png'
    fig.savefig(os.path.join(path, out_name), bbox_inches='tight', dpi=150)
    plt.close(fig)


# =====================================================
# Standalone funnel density view
# =====================================================
def funnel_joint_logprob(x1_grid, r_grid, d, scale1):
    X1, R = np.meshgrid(x1_grid, r_grid, indexing="ij")
    log_px1 = norm(loc=0, scale=scale1).logpdf(X1)

    k = d - 1
    log_pr_given_x1 = (
        (k - 1) * np.log(R)
        - (R**2) / (2.0 * np.exp(X1))
        - (k / 2.0) * np.log(2.0)
        - (k / 2.0) * X1
        - np.log(gamma(k / 2.0))
    )
    return log_px1 + log_pr_given_x1


def plot_funnel_density_only(
        sigma=3.0,
        dim=2,
        figsize=(8, 8),
        dpi=160,
        n_grid=400,
        log_floor=-20.0,
        output_path=None,
        divide_variance_by_two=True
    ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    funnel = Funnel(
        dim=dim,
        sigma=sigma,
        use_gpu=(device.type == 'cuda'),
        divide_variance_by_two=divide_variance_by_two
    )
    
    # x1 range
    X1_MIN, X1_MAX = max(-10, -10*1/np.sqrt(sigma)), 10 * np.sqrt(sigma)

    # new x2 range
    X2_MIN, X2_MAX = funnel_axis_limits(sigma, dim, divide_variance_by_two)

    x1_lin, x2_lin, logp = compute_log_joint_grid(
        funnel, 
        (X1_MIN, X1_MAX), 
        (X2_MIN, X2_MAX), 
        device=device,
        n1=n_grid, 
        n2=n_grid
    )
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    ax.set_facecolor("black")
    fig.patch.set_facecolor("white")
    cmap = plt.cm.magma.copy()
    cmap.set_under("black")
    
    vmax = float(np.max(logp))
    
    im = ax.imshow(
        logp,
        origin="lower",
        extent=[X2_MIN, X2_MAX, X1_MIN, X1_MAX],
        aspect="auto",
        cmap=cmap,
        vmin=log_floor,
        vmax=vmax,
    )

    
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(im, cax=cax)
    
    def log_to_prob(x, pos):
        return f'{np.exp(x):.1e}'
    
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(log_to_prob))
    cbar.ax.tick_params(color='black', labelcolor='black', labelsize=10)

    ax.set_xlabel(r"$x_2$", color="black", fontsize=14)
    ax.set_ylabel(r"$x_1$", color="black", fontsize=14)
    ax.set_xlim(X2_MIN, X2_MAX)
    ax.set_ylim(X1_MIN, X1_MAX)
    ax.tick_params(colors='black', labelsize=10)

    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    
    ax.set_title(
        f"Funnel Distribution (sig={sigma:.1f})",
        color='black', fontsize=16, pad=15
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', facecolor='white')
        print(f"Saved funnel density plot to {output_path}")
        plt.close()
    
    return fig, ax
