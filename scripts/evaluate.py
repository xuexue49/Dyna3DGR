#!/usr/bin/env python3
"""
Batch evaluation script for Dyna3DGR.

This script evaluates trained models on multiple patients and
generates comprehensive reports with metrics and visualizations.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import json
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.models import Gaussian3D, DeformationNetwork
from dyna3dgr.rendering import Medical2DSliceRenderer
from dyna3dgr.data import get_patient_ids, create_patient_dataloader
from dyna3dgr.utils.metrics import (
    compute_all_metrics,
    compute_sequence_metrics,
    compute_temporal_consistency,
    MetricsTracker,
    print_metrics,
)
from dyna3dgr.utils.visualization import (
    compare_slices,
    visualize_all_slices,
    create_comparison_grid,
    plot_metrics_over_time,
)


def load_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    gaussians = Gaussian3D(num_points=checkpoint.get('num_gaussians', 5000))
    deformation_net = DeformationNetwork()
    
    gaussians.load_state_dict(checkpoint['gaussians_state'])
    deformation_net.load_state_dict(checkpoint['deformation_net_state'])
    
    gaussians = gaussians.to(device)
    deformation_net = deformation_net.to(device)
    
    gaussians.eval()
    deformation_net.eval()
    
    return gaussians, deformation_net


def render_patient(
    gaussians: Gaussian3D,
    deformation_net: DeformationNetwork,
    renderer: Medical2DSliceRenderer,
    num_frames: int,
    device: str = 'cuda',
) -> np.ndarray:
    """Render complete patient sequence."""
    timestamps = torch.linspace(0, 1, num_frames, device=device)
    
    with torch.no_grad():
        rendered = renderer.render_with_time(
            means=gaussians.xyz,
            scales=gaussians.scale,
            rotations=gaussians.rotation,
            opacities=gaussians.opacity,
            features=gaussians.features,
            timestamps=timestamps,
            deformation_net=deformation_net,
        )
    
    return rendered.cpu().numpy()


def load_ground_truth(patient_dir: str, image_size: tuple) -> np.ndarray:
    """Load ground truth data."""
    from dyna3dgr.data import PatientDataset
    
    dataset = PatientDataset(
        patient_dir=patient_dir,
        image_size=image_size,
        load_segmentation=False,
        normalize=True,
    )
    
    frames = []
    for i in range(len(dataset)):
        sample = dataset[i]
        frames.append(sample['image'].numpy())
    
    ground_truth = np.stack(frames, axis=0)
    
    if ground_truth.ndim == 4:
        ground_truth = ground_truth.transpose(0, 3, 1, 2)
    else:
        ground_truth = ground_truth[:, np.newaxis, :, :]
    
    return ground_truth


def evaluate_patient(
    patient_id: str,
    checkpoint_path: str,
    patient_dir: str,
    renderer: Medical2DSliceRenderer,
    output_dir: Path,
    device: str = 'cuda',
    image_size: tuple = (256, 256),
    create_visualizations: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a single patient.
    
    Args:
        patient_id: Patient identifier
        checkpoint_path: Path to patient checkpoint
        patient_dir: Path to patient data
        renderer: Renderer
        output_dir: Output directory
        device: Device
        image_size: Image size
        create_visualizations: Whether to create visualizations
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    print(f"\nEvaluating {patient_id}...")
    
    # Load model
    gaussians, deformation_net = load_checkpoint(checkpoint_path, device)
    
    # Load ground truth
    ground_truth = load_ground_truth(patient_dir, image_size)
    num_frames = ground_truth.shape[0]
    
    # Render
    rendered = render_patient(
        gaussians,
        deformation_net,
        renderer,
        num_frames,
        device,
    )
    
    # Ensure same shape
    if rendered.shape != ground_truth.shape:
        print(f"Warning: Shape mismatch - rendered: {rendered.shape}, GT: {ground_truth.shape}")
        min_frames = min(rendered.shape[0], ground_truth.shape[0])
        min_slices = min(rendered.shape[1], ground_truth.shape[1])
        rendered = rendered[:min_frames, :min_slices]
        ground_truth = ground_truth[:min_frames, :min_slices]
    
    # Compute overall metrics
    overall_metrics = compute_all_metrics(rendered, ground_truth)
    
    # Compute temporal consistency
    overall_metrics['Temporal_Consistency_Rendered'] = compute_temporal_consistency(rendered)
    overall_metrics['Temporal_Consistency_GT'] = compute_temporal_consistency(ground_truth)
    
    # Compute per-frame metrics
    sequence_metrics = compute_sequence_metrics(rendered, ground_truth)
    
    # Add mean and std of sequence metrics
    for metric_name, values in sequence_metrics.items():
        overall_metrics[f'{metric_name}_Mean'] = np.mean(values)
        overall_metrics[f'{metric_name}_Std'] = np.std(values)
    
    # Save metrics
    patient_output_dir = output_dir / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(patient_output_dir / 'metrics.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        metrics_json = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in overall_metrics.items()}
        json.dump(metrics_json, f, indent=2)
    
    # Save sequence metrics
    sequence_df = pd.DataFrame(sequence_metrics)
    sequence_df.to_csv(patient_output_dir / 'sequence_metrics.csv', index=False)
    
    # Create visualizations
    if create_visualizations:
        print(f"Creating visualizations for {patient_id}...")
        
        # Comparison at middle time point
        mid_time = num_frames // 2
        mid_slice = rendered.shape[1] // 2
        
        fig = compare_slices(
            rendered,
            ground_truth,
            slice_idx=mid_slice,
            time_idx=mid_time,
            save_path=patient_output_dir / f'comparison_t{mid_time}_s{mid_slice}.png',
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        # All slices at middle time
        fig = visualize_all_slices(
            rendered,
            ground_truth,
            time_idx=mid_time,
            save_path=patient_output_dir / f'all_slices_t{mid_time}.png',
        )
        plt.close(fig)
        
        # Comparison grid
        fig = create_comparison_grid(
            rendered,
            ground_truth,
            num_samples=9,
            save_path=patient_output_dir / 'comparison_grid.png',
        )
        plt.close(fig)
        
        # Metrics over time
        fig = plot_metrics_over_time(
            {
                'PSNR': sequence_metrics['PSNR'],
                'SSIM': sequence_metrics['SSIM'],
                'MAE': sequence_metrics['MAE'],
            },
            save_path=patient_output_dir / 'metrics_over_time.png',
        )
        plt.close(fig)
    
    return overall_metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate Dyna3DGR models')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory containing patient data')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing patient checkpoints')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--image_size', type=int, nargs=2, default=[256, 256],
                        help='Image size (H W)')
    parser.add_argument('--num_slices', type=int, default=10,
                        help='Number of slices')
    parser.add_argument('--no_visualizations', action='store_true',
                        help='Skip creating visualizations')
    parser.add_argument('--patient_ids', type=str, nargs='+', default=None,
                        help='Specific patient IDs to evaluate (default: all in split)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = Path(args.checkpoint_dir)
    data_root = Path(args.data_root)
    
    # Get patient IDs
    if args.patient_ids:
        patient_ids = args.patient_ids
    else:
        patient_ids = get_patient_ids(str(data_root), args.split)
    
    print(f"Evaluating {len(patient_ids)} patients from {args.split} split")
    
    # Initialize renderer
    renderer = Medical2DSliceRenderer(
        image_size=tuple(args.image_size),
        num_slices=args.num_slices,
    ).to(device)
    
    # Evaluate each patient
    all_metrics = []
    metrics_tracker = MetricsTracker()
    
    for patient_id in tqdm(patient_ids, desc="Evaluating patients"):
        # Find checkpoint
        checkpoint_path = checkpoint_dir / patient_id / 'best.pth'
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / patient_id / 'final.pth'
        
        if not checkpoint_path.exists():
            print(f"Warning: No checkpoint found for {patient_id}, skipping")
            continue
        
        # Patient data directory
        patient_dir = data_root / patient_id
        if not patient_dir.exists():
            print(f"Warning: Patient directory not found: {patient_dir}, skipping")
            continue
        
        try:
            # Evaluate
            metrics = evaluate_patient(
                patient_id=patient_id,
                checkpoint_path=str(checkpoint_path),
                patient_dir=str(patient_dir),
                renderer=renderer,
                output_dir=output_dir,
                device=str(device),
                image_size=tuple(args.image_size),
                create_visualizations=not args.no_visualizations,
            )
            
            # Add patient ID to metrics
            metrics['Patient_ID'] = patient_id
            all_metrics.append(metrics)
            
            # Update tracker
            metrics_tracker.update({k: v for k, v in metrics.items() 
                                   if isinstance(v, (int, float))})
            
            # Print patient metrics
            print(f"\nMetrics for {patient_id}:")
            print_metrics({k: v for k, v in metrics.items() 
                          if isinstance(v, (int, float))})
        
        except Exception as e:
            print(f"Error evaluating {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save aggregate results
    if all_metrics:
        # Create DataFrame
        df = pd.DataFrame(all_metrics)
        
        # Reorder columns
        cols = ['Patient_ID'] + [col for col in df.columns if col != 'Patient_ID']
        df = df[cols]
        
        # Save to CSV
        df.to_csv(output_dir / 'all_metrics.csv', index=False)
        
        # Compute statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        stats_df = df[numeric_cols].describe()
        stats_df.to_csv(output_dir / 'statistics.csv')
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"\nEvaluated {len(all_metrics)} patients")
        print("\nAverage Metrics:")
        avg_metrics = metrics_tracker.get_average()
        print_metrics(avg_metrics)
        
        # Save summary
        with open(output_dir / 'summary.json', 'w') as f:
            summary = {
                'num_patients': len(all_metrics),
                'split': args.split,
                'average_metrics': avg_metrics,
            }
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to {output_dir}")
        print(f"- all_metrics.csv: Per-patient metrics")
        print(f"- statistics.csv: Statistical summary")
        print(f"- summary.json: Evaluation summary")
        print(f"- {len(all_metrics)} patient directories with detailed results")
    
    else:
        print("\nNo patients were successfully evaluated!")


if __name__ == '__main__':
    main()
