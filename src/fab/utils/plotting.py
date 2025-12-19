from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import itertools

from fab.types_ import LogProbFunc, Distribution


def plot_history(history):
    """Agnostic history plotter for quickly plotting a dictionary of logging info."""
    figure, axs = plt.subplots(len(history), 1, figsize=(7, 3*len(history.keys())))
    if len(history.keys()) == 1:
        axs = [axs]  # make iterable
    elif len(history.keys()) == 0:
        return
    for i, key in enumerate(history):
        data = pd.Series(history[key])
        data.replace([np.inf, -np.inf], np.nan, inplace=True)
        if sum(data.isna()) > 0:
            data = data.dropna()
            print(f"NaN encountered in {key} history")
        axs[i].plot(data)
        axs[i].set_title(key)
    plt.tight_layout()


def plot_contours(log_prob_func: LogProbFunc,
                  ax: Optional[plt.Axes] = None,
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  grid_width_n_points: int = 20,
                  n_contour_levels: Optional[int] = None,
                  log_prob_min: float = -1000.0):
    """Plot contours of a log_prob_func that is defined on 2D"""
    if ax is None:
        fig, ax = plt.subplots(1)
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)))
    log_p_x = log_prob_func(x_points).detach()
    log_p_x = torch.clamp_min(log_p_x, log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    x_points_dim1 = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2 = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    if n_contour_levels:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x, levels=n_contour_levels)
    else:
        ax.contour(x_points_dim1, x_points_dim2, log_p_x)


def plot_contours_mixture(log_prob_func: LogProbFunc,
                          ax: Optional[plt.Axes] = None,
                          bounds: Tuple[float, float] = (-5.0, 5.0),
                          grid_width_n_points: int = 200,
                          n_contour_levels: int = 20,
                          log_prob_min: Optional[float] = None):
    """Plot contours of a log_prob_func for mixture distributions.
    
    Automatically scales log probabilities to handle sharp peaks and valleys.
    Works well with mixtures of twisted Gaussians and other complex distributions.
    
    Args:
        log_prob_func: Function that computes log probability
        ax: Matplotlib axis to plot on
        bounds: Tuple of (min, max) for both dimensions
        grid_width_n_points: Resolution of the grid
        n_contour_levels: Number of contour levels to draw
        log_prob_min: Minimum log probability to display (auto-scaled if None)
    """
    if ax is None:
        fig, ax = plt.subplots(1)
    
    # Create grid points
    x_points_dim1 = torch.linspace(bounds[0], bounds[1], grid_width_n_points)
    x_points_dim2 = x_points_dim1
    x_points = torch.tensor(list(itertools.product(x_points_dim1, x_points_dim2)), dtype=torch.float32)
    
    # Compute log probabilities
    log_p_x = log_prob_func(x_points).detach()
    
    # Auto-scale: use the 10th percentile as the minimum if not provided
    if log_prob_min is None:
        log_prob_min = float(torch.quantile(log_p_x[~torch.isinf(log_p_x)], 0.1))
    
    # Clamp log probabilities for better visualization
    log_p_x = torch.clamp(log_p_x, min=log_prob_min)
    log_p_x = log_p_x.reshape((grid_width_n_points, grid_width_n_points))
    
    # Reshape grid for contour plot
    x_points_dim1_grid = x_points[:, 0].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    x_points_dim2_grid = x_points[:, 1].reshape((grid_width_n_points, grid_width_n_points)).numpy()
    log_p_x_np = log_p_x.numpy()
    
    # Plot contours with filled contours for better visibility
    contourf = ax.contourf(x_points_dim1_grid, x_points_dim2_grid, log_p_x_np, 
                           levels=n_contour_levels, cmap='viridis', alpha=0.6)
    ax.contour(x_points_dim1_grid, x_points_dim2_grid, log_p_x_np, 
              levels=n_contour_levels, colors='black', linewidths=0.5, alpha=0.3)
    
    return contourf


def plot_marginal_pair(samples: torch.Tensor,
                  ax: Optional[plt.Axes] = None,
                  marginal_dims: Tuple[int, int] = (0, 1),
                  bounds: Tuple[float, float] = (-5.0, 5.0),
                  alpha: float = 0.5):
    """Plot samples from marginal of distribution for a given pair of dimensions."""
    if not ax:
        fig, ax = plt.subplots(1)
    samples = torch.clamp(samples, bounds[0], bounds[1])
    samples = samples.cpu().detach()
    ax.plot(samples[:, marginal_dims[0]], samples[:, marginal_dims[1]], "o", alpha=alpha)
