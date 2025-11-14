#!/usr/bin/env python3
"""
Interactive visualization tool for comparing rendered results with ground truth.

This script provides an interactive viewer for exploring rendered slices,
comparing with ground truth, and analyzing reconstruction quality.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import nibabel as nib

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import Gaussian3D, DeformationNetwork
from dyna3dgr.rendering import Medical2DSliceRenderer
from dyna3dgr.data import create_patient_dataloader
from dyna3dgr.utils.visualization import (
    compare_slices,
    visualize_sequence,
    visualize_all_slices,
    create_comparison_grid,
    create_video_comparison,
)


class InteractiveViewer:
    """Interactive viewer for rendered results."""
    
    def __init__(
        self,
        rendered: np.ndarray,
        ground_truth: np.ndarray,
        patient_id: str = "unknown",
    ):
        """
        Initialize interactive viewer.
        
        Args:
            rendered: Rendered sequence [T, num_slices, H, W]
            ground_truth: Ground truth sequence [T, num_slices, H, W]
            patient_id: Patient identifier
        """
        self.rendered = rendered
        self.ground_truth = ground_truth
        self.patient_id = patient_id
        
        self.T, self.num_slices, self.H, self.W = rendered.shape
        
        self.current_time = 0
        self.current_slice = self.num_slices // 2
        
        self._setup_figure()
    
    def _setup_figure(self):
        """Setup matplotlib figure and widgets."""
        self.fig = plt.figure(figsize=(16, 8))
        
        # Create subplots
        gs = self.fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        self.ax_gt = self.fig.add_subplot(gs[0, 0])
        self.ax_rendered = self.fig.add_subplot(gs[0, 1])
        self.ax_diff = self.fig.add_subplot(gs[0, 2])
        self.ax_profile = self.fig.add_subplot(gs[1, :])
        
        # Initial display
        self._update_display()
        
        # Add sliders
        ax_time = plt.axes([0.15, 0.02, 0.3, 0.03])
        ax_slice = plt.axes([0.6, 0.02, 0.3, 0.03])
        
        self.slider_time = Slider(
            ax_time, 'Time', 0, self.T - 1,
            valinit=self.current_time,
            valstep=1
        )
        self.slider_slice = Slider(
            ax_slice, 'Slice', 0, self.num_slices - 1,
            valinit=self.current_slice,
            valstep=1
        )
        
        self.slider_time.on_changed(self._on_time_change)
        self.slider_slice.on_changed(self._on_slice_change)
        
        # Add buttons
        ax_prev = plt.axes([0.15, 0.08, 0.1, 0.04])
        ax_next = plt.axes([0.26, 0.08, 0.1, 0.04])
        ax_play = plt.axes([0.6, 0.08, 0.1, 0.04])
        ax_save = plt.axes([0.71, 0.08, 0.1, 0.04])
        
        self.btn_prev = Button(ax_prev, 'Previous')
        self.btn_next = Button(ax_next, 'Next')
        self.btn_play = Button(ax_play, 'Play')
        self.btn_save = Button(ax_save, 'Save')
        
        self.btn_prev.on_clicked(self._on_prev)
        self.btn_next.on_clicked(self._on_next)
        self.btn_play.on_clicked(self._on_play)
        self.btn_save.on_clicked(self._on_save)
        
        self.playing = False
        self.timer = None
    
    def _update_display(self):
        """Update all displays."""
        t = self.current_time
        s = self.current_slice
        
        gt_slice = self.ground_truth[t, s]
        rend_slice = self.rendered[t, s]
        diff = np.abs(rend_slice - gt_slice)
        
        # Ground truth
        self.ax_gt.clear()
        self.ax_gt.imshow(gt_slice, cmap='gray')
        self.ax_gt.set_title('Ground Truth', fontweight='bold')
        self.ax_gt.axis('off')
        
        # Rendered
        self.ax_rendered.clear()
        self.ax_rendered.imshow(rend_slice, cmap='gray')
        self.ax_rendered.set_title('Rendered', fontweight='bold')
        self.ax_rendered.axis('off')
        
        # Difference
        self.ax_diff.clear()
        im_diff = self.ax_diff.imshow(diff, cmap='hot')
        self.ax_diff.set_title('Absolute Difference', fontweight='bold')
        self.ax_diff.axis('off')
        plt.colorbar(im_diff, ax=self.ax_diff, fraction=0.046)
        
        # Statistics
        mae = np.mean(diff)
        mse = np.mean(diff ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        self.ax_diff.text(
            0.5, -0.1,
            f'MAE: {mae:.4f}  MSE: {mse:.4f}  PSNR: {psnr:.2f} dB',
            transform=self.ax_diff.transAxes,
            ha='center',
            fontsize=10,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        # Profile plot
        self.ax_profile.clear()
        center_row = self.H // 2
        gt_profile = gt_slice[center_row, :]
        rend_profile = rend_slice[center_row, :]
        
        self.ax_profile.plot(gt_profile, label='Ground Truth', linewidth=2)
        self.ax_profile.plot(rend_profile, label='Rendered', linewidth=2, linestyle='--')
        self.ax_profile.set_xlabel('Pixel Position', fontsize=12)
        self.ax_profile.set_ylabel('Intensity', fontsize=12)
        self.ax_profile.set_title(f'Center Row Profile (Row {center_row})', fontweight='bold')
        self.ax_profile.legend()
        self.ax_profile.grid(True, alpha=0.3)
        
        # Update title
        self.fig.suptitle(
            f'Patient: {self.patient_id} | Time: {t}/{self.T-1} | Slice: {s}/{self.num_slices-1}',
            fontsize=14,
            fontweight='bold'
        )
        
        self.fig.canvas.draw_idle()
    
    def _on_time_change(self, val):
        """Handle time slider change."""
        self.current_time = int(val)
        self._update_display()
    
    def _on_slice_change(self, val):
        """Handle slice slider change."""
        self.current_slice = int(val)
        self._update_display()
    
    def _on_prev(self, event):
        """Go to previous time frame."""
        if self.current_time > 0:
            self.current_time -= 1
            self.slider_time.set_val(self.current_time)
    
    def _on_next(self, event):
        """Go to next time frame."""
        if self.current_time < self.T - 1:
            self.current_time += 1
            self.slider_time.set_val(self.current_time)
    
    def _on_play(self, event):
        """Toggle play/pause."""
        self.playing = not self.playing
        
        if self.playing:
            self.btn_play.label.set_text('Pause')
            self._play_animation()
        else:
            self.btn_play.label.set_text('Play')
            if self.timer:
                self.timer.stop()
    
    def _play_animation(self):
        """Play animation."""
        def update(frame):
            if self.playing:
                self.current_time = (self.current_time + 1) % self.T
                self.slider_time.set_val(self.current_time)
        
        from matplotlib.animation import FuncAnimation
        self.timer = FuncAnimation(
            self.fig,
            update,
            interval=200,
            repeat=True
        )
    
    def _on_save(self, event):
        """Save current view."""
        output_dir = Path('visualizations') / self.patient_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f't{self.current_time:03d}_s{self.current_slice:02d}.png'
        filepath = output_dir / filename
        
        self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved to {filepath}")
    
    def show(self):
        """Show the viewer."""
        plt.show()


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """
    Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        gaussians: Gaussian3D model
        deformation_net: DeformationNetwork model
        config: Configuration dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Initialize models
    gaussians = Gaussian3D(num_points=checkpoint.get('num_gaussians', 5000))
    deformation_net = DeformationNetwork()
    
    # Load states
    gaussians.load_state_dict(checkpoint['gaussians_state'])
    deformation_net.load_state_dict(checkpoint['deformation_net_state'])
    
    gaussians = gaussians.to(device)
    deformation_net = deformation_net.to(device)
    
    gaussians.eval()
    deformation_net.eval()
    
    config = checkpoint.get('config', {})
    
    return gaussians, deformation_net, config


def render_patient_sequence(
    gaussians: Gaussian3D,
    deformation_net: DeformationNetwork,
    renderer: Medical2DSliceRenderer,
    num_frames: int,
    device: str = 'cuda',
) -> np.ndarray:
    """
    Render complete patient sequence.
    
    Args:
        gaussians: Gaussian3D model
        deformation_net: DeformationNetwork model
        renderer: Renderer
        num_frames: Number of time frames
        device: Device
    
    Returns:
        rendered_sequence: Rendered sequence [T, num_slices, H, W]
    """
    timestamps = torch.linspace(0, 1, num_frames, device=device)
    
    with torch.no_grad():
        rendered_sequence = renderer.render_with_time(
            means=gaussians.xyz,
            scales=gaussians.scale,
            rotations=gaussians.rotation,
            opacities=gaussians.opacity,
            features=gaussians.features,
            timestamps=timestamps,
            deformation_net=deformation_net,
        )
    
    return rendered_sequence.cpu().numpy()


def load_ground_truth(patient_dir: str, image_size: tuple) -> np.ndarray:
    """
    Load ground truth data.
    
    Args:
        patient_dir: Patient directory
        image_size: Target image size
    
    Returns:
        ground_truth: Ground truth sequence [T, num_slices, H, W]
    """
    from dyna3dgr.data import PatientDataset
    
    dataset = PatientDataset(
        patient_dir=patient_dir,
        image_size=image_size,
        load_segmentation=False,
        normalize=True,
    )
    
    # Load all frames
    frames = []
    for i in range(len(dataset)):
        sample = dataset[i]
        frames.append(sample['image'].numpy())
    
    # Stack to [T, H, W] or [T, H, W, D]
    ground_truth = np.stack(frames, axis=0)
    
    # If 3D, assume format is [T, H, W, D], transpose to [T, D, H, W]
    if ground_truth.ndim == 4:
        ground_truth = ground_truth.transpose(0, 3, 1, 2)
    else:
        # If 2D, add slice dimension
        ground_truth = ground_truth[:, np.newaxis, :, :]
    
    return ground_truth


def main():
    parser = argparse.ArgumentParser(description='Visualize rendered results')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--patient_dir', type=str, required=True,
                        help='Path to patient data directory')
    parser.add_argument('--output_dir', type=str, default='visualizations',
                        help='Output directory for saved visualizations')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (H W)')
    parser.add_argument('--num_slices', type=int, default=10,
                        help='Number of slices')
    parser.add_argument('--num_frames', type=int, default=20,
                        help='Number of time frames')
    parser.add_argument('--mode', type=str, default='interactive',
                        choices=['interactive', 'batch', 'video', 'grid'],
                        help='Visualization mode')
    parser.add_argument('--slice_idx', type=int, default=None,
                        help='Specific slice index to visualize')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    patient_id = Path(args.patient_dir).name
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    gaussians, deformation_net, config = load_checkpoint(args.checkpoint, device)
    
    print(f"Initializing renderer...")
    renderer = Medical2DSliceRenderer(
        image_size=tuple(args.image_size),
        num_slices=args.num_slices,
    ).to(device)
    
    print(f"Rendering sequence...")
    rendered = render_patient_sequence(
        gaussians,
        deformation_net,
        renderer,
        args.num_frames,
        device,
    )
    
    print(f"Loading ground truth from {args.patient_dir}...")
    ground_truth = load_ground_truth(args.patient_dir, tuple(args.image_size))
    
    # Ensure same number of frames
    if ground_truth.shape[0] != rendered.shape[0]:
        print(f"Warning: GT has {ground_truth.shape[0]} frames, rendered has {rendered.shape[0]} frames")
        min_frames = min(ground_truth.shape[0], rendered.shape[0])
        ground_truth = ground_truth[:min_frames]
        rendered = rendered[:min_frames]
    
    print(f"Rendered shape: {rendered.shape}")
    print(f"Ground truth shape: {ground_truth.shape}")
    
    # Visualization
    if args.mode == 'interactive':
        print("Starting interactive viewer...")
        viewer = InteractiveViewer(rendered, ground_truth, patient_id)
        viewer.show()
    
    elif args.mode == 'batch':
        print("Creating batch visualizations...")
        patient_output_dir = output_dir / patient_id
        patient_output_dir.mkdir(exist_ok=True)
        
        # Visualize all slices at middle time point
        mid_time = rendered.shape[0] // 2
        fig = visualize_all_slices(
            rendered,
            ground_truth,
            time_idx=mid_time,
            save_path=patient_output_dir / f'all_slices_t{mid_time}.png'
        )
        plt.close(fig)
        
        # Visualize sequence for middle slice
        mid_slice = args.slice_idx if args.slice_idx is not None else rendered.shape[1] // 2
        anim = visualize_sequence(
            rendered,
            ground_truth,
            slice_idx=mid_slice,
            save_path=patient_output_dir / f'sequence_s{mid_slice}.gif'
        )
        plt.close()
        
        # Create comparison grid
        fig = create_comparison_grid(
            rendered,
            ground_truth,
            num_samples=9,
            save_path=patient_output_dir / 'comparison_grid.png'
        )
        plt.close(fig)
        
        print(f"Batch visualizations saved to {patient_output_dir}")
    
    elif args.mode == 'video':
        print("Creating video...")
        slice_idx = args.slice_idx if args.slice_idx is not None else rendered.shape[1] // 2
        video_path = output_dir / patient_id / f'comparison_s{slice_idx}.mp4'
        video_path.parent.mkdir(exist_ok=True)
        
        create_video_comparison(
            rendered,
            ground_truth,
            slice_idx=slice_idx,
            output_path=str(video_path),
            fps=10
        )
    
    elif args.mode == 'grid':
        print("Creating comparison grid...")
        fig = create_comparison_grid(
            rendered,
            ground_truth,
            num_samples=16,
            save_path=output_dir / patient_id / 'grid.png'
        )
        plt.show()


if __name__ == '__main__':
    main()
