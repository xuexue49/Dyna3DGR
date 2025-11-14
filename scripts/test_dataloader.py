#!/usr/bin/env python3
"""
Test script for ACDC data loader.

This script tests the ACDC data loader and visualizes sample data.
"""

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dyna3dgr.data import create_acdc_dataloader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test ACDC data loader')
    
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to ACDC data root'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        choices=['train', 'val', 'test'],
        help='Dataset split'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=2,
        help='Batch size'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of samples to visualize'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/dataloader_test',
        help='Output directory for visualizations'
    )
    
    return parser.parse_args()


def visualize_sample(data, sample_idx, output_dir):
    """
    Visualize a data sample.
    
    Args:
        data: Data dictionary
        sample_idx: Sample index in batch
        output_dir: Output directory
    """
    images = data['images'][sample_idx].numpy()  # [T, H, W, D]
    timestamps = data['timestamps'][sample_idx].numpy()  # [T]
    patient_id = data['patient_ids'][sample_idx]
    
    # Get actual sequence length
    length = data['lengths'][sample_idx].item()
    images = images[:length]
    timestamps = timestamps[:length]
    
    print(f"\nPatient: {patient_id}")
    print(f"Sequence length: {length}")
    print(f"Image shape: {images.shape}")
    print(f"Timestamps: {timestamps}")
    
    # Visualize middle slice of each frame
    num_frames = min(length, 8)  # Show at most 8 frames
    frame_indices = np.linspace(0, length - 1, num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, num_frames, figsize=(num_frames * 3, 6))
    
    if num_frames == 1:
        axes = axes.reshape(-1, 1)
    
    for i, frame_idx in enumerate(frame_indices):
        frame = images[frame_idx]  # [H, W, D]
        
        # Get middle slice
        mid_slice = frame.shape[2] // 2
        image_slice = frame[:, :, mid_slice]
        
        # Plot image
        axes[0, i].imshow(image_slice, cmap='gray')
        axes[0, i].set_title(f't={timestamps[frame_idx]:.2f}')
        axes[0, i].axis('off')
        
        # Plot segmentation if available
        if 'segmentations' in data:
            seg = data['segmentations'][sample_idx, frame_idx].numpy()
            seg_slice = seg[:, :, mid_slice]
            
            axes[1, i].imshow(seg_slice, cmap='tab10', vmin=0, vmax=3)
            axes[1, i].set_title(f'Segmentation')
            axes[1, i].axis('off')
        else:
            axes[1, i].axis('off')
    
    plt.suptitle(f'Patient: {patient_id}')
    plt.tight_layout()
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{patient_id}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {output_file}")
    
    plt.close()


def print_batch_info(data):
    """Print information about a batch."""
    print("\n" + "="*60)
    print("Batch Information")
    print("="*60)
    
    print(f"Batch size: {data['images'].shape[0]}")
    print(f"Image shape: {data['images'].shape}")
    print(f"Timestamps shape: {data['timestamps'].shape}")
    print(f"Lengths: {data['lengths'].tolist()}")
    
    if 'segmentations' in data:
        print(f"Segmentations shape: {data['segmentations'].shape}")
    
    print(f"Patient IDs: {data['patient_ids']}")
    
    # Print metadata for first sample
    if len(data['metadata']) > 0:
        print(f"\nMetadata (first sample):")
        metadata = data['metadata'][0]
        for key, value in metadata.items():
            if key != 'info':
                print(f"  {key}: {value}")


def main():
    """Main function."""
    args = parse_args()
    
    print("Creating ACDC data loader...")
    print(f"Data root: {args.data_root}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    
    # Create data loader
    try:
        dataloader = create_acdc_dataloader(
            data_root=args.data_root,
            split=args.split,
            batch_size=args.batch_size,
            num_workers=0,  # Use 0 for debugging
            image_size=(256, 256),
            num_frames=None,  # Load all frames
            normalize=True,
            augmentation=False,
            load_segmentation=True,
        )
    except Exception as e:
        print(f"Error creating data loader: {e}")
        return
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test loading a few batches
    print("\nLoading and visualizing samples...")
    
    num_visualized = 0
    
    for batch_idx, data in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}/{len(dataloader)}")
        
        # Print batch info
        print_batch_info(data)
        
        # Visualize samples
        batch_size = data['images'].shape[0]
        for sample_idx in range(batch_size):
            if num_visualized >= args.num_samples:
                break
            
            visualize_sample(data, sample_idx, args.output_dir)
            num_visualized += 1
        
        if num_visualized >= args.num_samples:
            break
    
    print("\n" + "="*60)
    print("Testing completed successfully!")
    print("="*60)
    print(f"Visualizations saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
