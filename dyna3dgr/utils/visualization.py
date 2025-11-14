"""
Visualization utilities for Dyna3DGR.

This module provides functions for visualizing rendered results,
comparing with ground truth, and analyzing motion fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import torch
from typing import Optional, Tuple, List, Dict
from pathlib import Path
import cv2


def compare_slices(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    slice_idx: int = 0,
    time_idx: int = 0,
    save_path: Optional[str] = None,
    show_difference: bool = True,
    colormap: str = 'gray',
) -> plt.Figure:
    """
    Compare rendered slice with ground truth.
    
    Args:
        rendered: Rendered slices [T, num_slices, H, W] or [num_slices, H, W]
        ground_truth: Ground truth slices [T, num_slices, H, W] or [num_slices, H, W]
        slice_idx: Index of slice to visualize
        time_idx: Index of time point (if 4D)
        save_path: Path to save figure
        show_difference: Whether to show difference map
        colormap: Colormap for visualization
    
    Returns:
        fig: Matplotlib figure
    """
    # Handle dimensions
    if rendered.ndim == 4:
        rendered_slice = rendered[time_idx, slice_idx]
        gt_slice = ground_truth[time_idx, slice_idx]
    elif rendered.ndim == 3:
        rendered_slice = rendered[slice_idx]
        gt_slice = ground_truth[slice_idx]
    else:
        rendered_slice = rendered
        gt_slice = ground_truth
    
    # Create figure
    if show_difference:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground truth
    im0 = axes[0].imshow(gt_slice, cmap=colormap)
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    
    # Rendered
    im1 = axes[1].imshow(rendered_slice, cmap=colormap)
    axes[1].set_title('Rendered', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Difference
    if show_difference:
        diff = np.abs(rendered_slice - gt_slice)
        im2 = axes[2].imshow(diff, cmap='hot')
        axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        plt.colorbar(im2, ax=axes[2], fraction=0.046)
        
        # Add statistics
        mae = np.mean(diff)
        mse = np.mean(diff ** 2)
        axes[2].text(
            0.5, -0.1,
            f'MAE: {mae:.4f}  MSE: {mse:.4f}',
            transform=axes[2].transAxes,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
    
    plt.suptitle(
        f'Slice {slice_idx}, Time {time_idx}',
        fontsize=16,
        fontweight='bold',
        y=1.02
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_sequence(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    slice_idx: int = 0,
    save_path: Optional[str] = None,
    interval: int = 200,
) -> animation.FuncAnimation:
    """
    Create animation comparing rendered and ground truth sequences.
    
    Args:
        rendered: Rendered sequence [T, num_slices, H, W]
        ground_truth: Ground truth sequence [T, num_slices, H, W]
        slice_idx: Index of slice to visualize
        save_path: Path to save animation (e.g., 'output.gif')
        interval: Delay between frames in milliseconds
    
    Returns:
        anim: Matplotlib animation object
    """
    T = rendered.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Initialize images
    rendered_slice = rendered[0, slice_idx]
    gt_slice = ground_truth[0, slice_idx]
    diff = np.abs(rendered_slice - gt_slice)
    
    im0 = axes[0].imshow(gt_slice, cmap='gray', animated=True)
    im1 = axes[1].imshow(rendered_slice, cmap='gray', animated=True)
    im2 = axes[2].imshow(diff, cmap='hot', animated=True)
    
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].set_title('Rendered', fontsize=14, fontweight='bold')
    axes[2].set_title('Difference', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.axis('off')
    
    plt.colorbar(im0, ax=axes[0], fraction=0.046)
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    plt.colorbar(im2, ax=axes[2], fraction=0.046)
    
    # Time text
    time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=12, fontweight='bold')
    
    def update(frame):
        rendered_slice = rendered[frame, slice_idx]
        gt_slice = ground_truth[frame, slice_idx]
        diff = np.abs(rendered_slice - gt_slice)
        
        im0.set_array(gt_slice)
        im1.set_array(rendered_slice)
        im2.set_array(diff)
        
        time_text.set_text(f'Frame {frame}/{T-1}')
        
        return [im0, im1, im2, time_text]
    
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=T,
        interval=interval,
        blit=True,
        repeat=True
    )
    
    if save_path:
        if save_path.endswith('.gif'):
            anim.save(save_path, writer='pillow', fps=1000//interval)
        elif save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
    
    return anim


def visualize_all_slices(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    time_idx: int = 0,
    save_path: Optional[str] = None,
    max_cols: int = 5,
) -> plt.Figure:
    """
    Visualize all slices at a given time point.
    
    Args:
        rendered: Rendered slices [T, num_slices, H, W] or [num_slices, H, W]
        ground_truth: Ground truth slices [T, num_slices, H, W] or [num_slices, H, W]
        time_idx: Time index (if 4D)
        save_path: Path to save figure
        max_cols: Maximum number of columns
    
    Returns:
        fig: Matplotlib figure
    """
    # Handle dimensions
    if rendered.ndim == 4:
        rendered_slices = rendered[time_idx]
        gt_slices = ground_truth[time_idx]
    else:
        rendered_slices = rendered
        gt_slices = ground_truth
    
    num_slices = rendered_slices.shape[0]
    num_cols = min(max_cols, num_slices)
    num_rows = (num_slices + num_cols - 1) // num_cols
    
    # Create figure with two rows (GT and rendered)
    fig, axes = plt.subplots(num_rows * 2, num_cols, figsize=(num_cols * 3, num_rows * 6))
    
    if num_rows == 1:
        axes = axes.reshape(2, num_cols)
    
    for i in range(num_slices):
        row = (i // num_cols) * 2
        col = i % num_cols
        
        # Ground truth
        axes[row, col].imshow(gt_slices[i], cmap='gray')
        axes[row, col].set_title(f'GT Slice {i}', fontsize=10)
        axes[row, col].axis('off')
        
        # Rendered
        axes[row + 1, col].imshow(rendered_slices[i], cmap='gray')
        axes[row + 1, col].set_title(f'Rendered Slice {i}', fontsize=10)
        axes[row + 1, col].axis('off')
    
    # Hide unused subplots
    for i in range(num_slices, num_rows * num_cols):
        row = (i // num_cols) * 2
        col = i % num_cols
        axes[row, col].axis('off')
        axes[row + 1, col].axis('off')
    
    plt.suptitle(f'All Slices at Time {time_idx}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def visualize_motion_field(
    motion_field: np.ndarray,
    slice_idx: int = 0,
    time_idx: int = 0,
    background: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    scale: float = 1.0,
    step: int = 8,
) -> plt.Figure:
    """
    Visualize motion field as arrows.
    
    Args:
        motion_field: Motion field [T, N, 3] or [N, 3]
        slice_idx: Slice index for visualization
        time_idx: Time index
        background: Background image [H, W]
        save_path: Path to save figure
        scale: Scale factor for arrows
        step: Step size for arrow grid
    
    Returns:
        fig: Matplotlib figure
    """
    # Handle dimensions
    if motion_field.ndim == 3:
        motion = motion_field[time_idx]
    else:
        motion = motion_field
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Show background if provided
    if background is not None:
        ax.imshow(background, cmap='gray', alpha=0.7)
    
    # Extract 2D motion (x, y components)
    # Assume motion is in format [N, 3] where columns are (x, y, z)
    x = motion[:, 0]
    y = motion[:, 1]
    dx = motion[:, 0]  # Displacement in x
    dy = motion[:, 1]  # Displacement in y
    
    # Create grid for visualization
    x_grid = x[::step]
    y_grid = y[::step]
    dx_grid = dx[::step]
    dy_grid = dy[::step]
    
    # Plot arrows
    ax.quiver(
        x_grid, y_grid, dx_grid, dy_grid,
        scale=scale,
        scale_units='xy',
        angles='xy',
        color='red',
        alpha=0.8,
        width=0.003
    )
    
    ax.set_title(f'Motion Field - Time {time_idx}', fontsize=14, fontweight='bold')
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_comparison_grid(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    num_samples: int = 9,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Create a grid showing multiple time points and slices.
    
    Args:
        rendered: Rendered sequence [T, num_slices, H, W]
        ground_truth: Ground truth sequence [T, num_slices, H, W]
        num_samples: Number of samples to show (will be arranged in grid)
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    T, num_slices, H, W = rendered.shape
    
    # Calculate grid size
    grid_size = int(np.ceil(np.sqrt(num_samples)))
    
    # Sample time points and slices
    time_indices = np.linspace(0, T-1, grid_size, dtype=int)
    slice_indices = np.linspace(0, num_slices-1, grid_size, dtype=int)
    
    # Create figure
    fig = plt.figure(figsize=(grid_size * 6, grid_size * 3))
    gs = GridSpec(grid_size, grid_size * 2, figure=fig, hspace=0.3, wspace=0.2)
    
    for i, t in enumerate(time_indices):
        for j, s in enumerate(slice_indices):
            # Ground truth
            ax_gt = fig.add_subplot(gs[i, j*2])
            ax_gt.imshow(ground_truth[t, s], cmap='gray')
            ax_gt.set_title(f'GT T={t} S={s}', fontsize=8)
            ax_gt.axis('off')
            
            # Rendered
            ax_rend = fig.add_subplot(gs[i, j*2+1])
            ax_rend.imshow(rendered[t, s], cmap='gray')
            ax_rend.set_title(f'Rendered T={t} S={s}', fontsize=8)
            ax_rend.axis('off')
    
    plt.suptitle('Comparison Grid', fontsize=16, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_metrics_over_time(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot metrics over time.
    
    Args:
        metrics: Dictionary of metric names to values over time
        save_path: Path to save figure
    
    Returns:
        fig: Matplotlib figure
    """
    num_metrics = len(metrics)
    fig, axes = plt.subplots(num_metrics, 1, figsize=(10, 4 * num_metrics))
    
    if num_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Time Frame', fontsize=12)
        ax.set_ylabel(name, fontsize=12)
        ax.set_title(f'{name} over Time', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.4f}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def create_video_comparison(
    rendered: np.ndarray,
    ground_truth: np.ndarray,
    slice_idx: int,
    output_path: str,
    fps: int = 10,
    codec: str = 'mp4v',
) -> None:
    """
    Create video comparing rendered and ground truth sequences.
    
    Args:
        rendered: Rendered sequence [T, num_slices, H, W]
        ground_truth: Ground truth sequence [T, num_slices, H, W]
        slice_idx: Slice to visualize
        output_path: Output video path
        fps: Frames per second
        codec: Video codec
    """
    T = rendered.shape[0]
    H, W = rendered.shape[2:4]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (W * 3, H))
    
    for t in range(T):
        # Get slices
        gt_slice = ground_truth[t, slice_idx]
        rend_slice = rendered[t, slice_idx]
        diff = np.abs(rend_slice - gt_slice)
        
        # Normalize to 0-255
        gt_slice = ((gt_slice - gt_slice.min()) / (gt_slice.max() - gt_slice.min() + 1e-8) * 255).astype(np.uint8)
        rend_slice = ((rend_slice - rend_slice.min()) / (rend_slice.max() - rend_slice.min() + 1e-8) * 255).astype(np.uint8)
        diff = ((diff - diff.min()) / (diff.max() - diff.min() + 1e-8) * 255).astype(np.uint8)
        
        # Convert to BGR
        gt_bgr = cv2.cvtColor(gt_slice, cv2.COLOR_GRAY2BGR)
        rend_bgr = cv2.cvtColor(rend_slice, cv2.COLOR_GRAY2BGR)
        diff_bgr = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
        
        # Add labels
        cv2.putText(gt_bgr, 'Ground Truth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(rend_bgr, 'Rendered', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(diff_bgr, 'Difference', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Concatenate horizontally
        frame = np.hstack([gt_bgr, rend_bgr, diff_bgr])
        
        # Write frame
        out.write(frame)
    
    out.release()
    print(f"Video saved to {output_path}")


def visualize_gaussian_distribution(
    means: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    slice_z: float = 0.0,
    save_path: Optional[str] = None,
    background: Optional[np.ndarray] = None,
) -> plt.Figure:
    """
    Visualize Gaussian distribution in 3D space.
    
    Args:
        means: Gaussian centers [N, 3]
        scales: Gaussian scales [N, 3]
        opacities: Gaussian opacities [N, 1]
        slice_z: Z-coordinate of slice to highlight
        save_path: Path to save figure
        background: Background image for slice
    
    Returns:
        fig: Matplotlib figure
    """
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(
        means[:, 0],
        means[:, 1],
        means[:, 2],
        c=opacities.flatten(),
        cmap='viridis',
        s=scales.mean(axis=1) * 10,
        alpha=0.6
    )
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Gaussian Distribution', fontweight='bold')
    plt.colorbar(scatter, ax=ax1, label='Opacity')
    
    # Highlight slice
    ax1.plot_surface(
        np.array([[means[:, 0].min(), means[:, 0].max()],
                  [means[:, 0].min(), means[:, 0].max()]]),
        np.array([[means[:, 1].min(), means[:, 1].min()],
                  [means[:, 1].max(), means[:, 1].max()]]),
        np.array([[slice_z, slice_z], [slice_z, slice_z]]),
        alpha=0.3,
        color='red'
    )
    
    # XY projection
    ax2 = fig.add_subplot(132)
    ax2.scatter(means[:, 0], means[:, 1], c=opacities.flatten(), cmap='viridis', s=10, alpha=0.6)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('XY Projection', fontweight='bold')
    ax2.set_aspect('equal')
    
    # XZ projection
    ax3 = fig.add_subplot(133)
    ax3.scatter(means[:, 0], means[:, 2], c=opacities.flatten(), cmap='viridis', s=10, alpha=0.6)
    ax3.axhline(slice_z, color='red', linestyle='--', label=f'Slice z={slice_z:.2f}')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.set_title('XZ Projection', fontweight='bold')
    ax3.legend()
    ax3.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
