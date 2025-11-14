#!/usr/bin/env python3
"""
Data Preprocessing Script for ACDC Dataset

This script preprocesses the ACDC dataset:
- Resamples images to uniform spacing
- Normalizes intensity values
- Crops/pads to consistent size
- Extracts point clouds from segmentation masks
- Saves preprocessed data
"""

import os
import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Preprocess ACDC dataset')
    
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing raw ACDC data'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for preprocessed data'
    )
    parser.add_argument(
        '--target_spacing',
        type=float,
        nargs=3,
        default=[1.5, 1.5, 10.0],
        help='Target voxel spacing (x, y, z) in mm'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        nargs=2,
        default=[256, 256],
        help='Target image size (H, W)'
    )
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize intensity values'
    )
    parser.add_argument(
        '--extract_points',
        action='store_true',
        help='Extract point clouds from segmentation masks'
    )
    parser.add_argument(
        '--num_points',
        type=int,
        default=10000,
        help='Number of points to sample from segmentation'
    )
    
    return parser.parse_args()


def load_nifti(file_path):
    """Load NIfTI file."""
    nii = nib.load(str(file_path))
    data = nii.get_fdata()
    affine = nii.affine
    header = nii.header
    
    return data, affine, header


def save_nifti(data, affine, header, file_path):
    """Save NIfTI file."""
    nii = nib.Nifti1Image(data, affine, header)
    nib.save(nii, str(file_path))


def resample_image(image, original_spacing, target_spacing, is_label=False):
    """
    Resample image to target spacing.
    
    Args:
        image: Input image array
        original_spacing: Original voxel spacing (x, y, z)
        target_spacing: Target voxel spacing (x, y, z)
        is_label: Whether this is a label image
    
    Returns:
        resampled: Resampled image
    """
    from scipy.ndimage import zoom
    
    # Calculate zoom factors
    zoom_factors = np.array(original_spacing) / np.array(target_spacing)
    
    # Resample
    order = 0 if is_label else 3  # Nearest for labels, cubic for images
    resampled = zoom(image, zoom_factors, order=order)
    
    return resampled


def normalize_intensity(image, percentile_clip=True):
    """
    Normalize image intensity.
    
    Args:
        image: Input image
        percentile_clip: Whether to clip outliers
    
    Returns:
        normalized: Normalized image
    """
    if percentile_clip:
        # Clip outliers
        p1, p99 = np.percentile(image, [1, 99])
        image = np.clip(image, p1, p99)
    
    # Normalize to [0, 1]
    image_min = image.min()
    image_max = image.max()
    
    if image_max > image_min:
        normalized = (image - image_min) / (image_max - image_min)
    else:
        normalized = np.zeros_like(image)
    
    return normalized


def crop_or_pad(image, target_size):
    """
    Crop or pad image to target size.
    
    Args:
        image: Input image [H, W, D]
        target_size: Target size (H, W)
    
    Returns:
        processed: Cropped/padded image
    """
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate padding/cropping
    if h < target_h:
        pad_h = (target_h - h) // 2
        pad_h_after = target_h - h - pad_h
    else:
        pad_h = 0
        pad_h_after = 0
    
    if w < target_w:
        pad_w = (target_w - w) // 2
        pad_w_after = target_w - w - pad_w
    else:
        pad_w = 0
        pad_w_after = 0
    
    # Pad if needed
    if pad_h > 0 or pad_w > 0:
        pad_width = [(pad_h, pad_h_after), (pad_w, pad_w_after)]
        if image.ndim == 3:
            pad_width.append((0, 0))
        image = np.pad(image, pad_width, mode='constant', constant_values=0)
    
    # Crop if needed
    h, w = image.shape[:2]
    if h > target_h or w > target_w:
        start_h = (h - target_h) // 2 if h > target_h else 0
        start_w = (w - target_w) // 2 if w > target_w else 0
        
        if image.ndim == 3:
            image = image[start_h:start_h+target_h, start_w:start_w+target_w, :]
        else:
            image = image[start_h:start_h+target_h, start_w:start_w+target_w]
    
    return image


def extract_point_cloud(segmentation, num_points=10000, labels=[1, 2, 3]):
    """
    Extract point cloud from segmentation mask.
    
    Args:
        segmentation: Segmentation mask [H, W, D]
        num_points: Number of points to sample
        labels: List of label values to include
    
    Returns:
        points: Point cloud [N, 3]
        point_labels: Label for each point [N]
    """
    # Get all foreground voxels
    mask = np.isin(segmentation, labels)
    coords = np.argwhere(mask)
    
    if len(coords) == 0:
        return np.zeros((0, 3)), np.zeros((0,), dtype=int)
    
    # Get labels for each coordinate
    point_labels = segmentation[coords[:, 0], coords[:, 1], coords[:, 2]]
    
    # Sample points
    if len(coords) > num_points:
        indices = np.random.choice(len(coords), num_points, replace=False)
        coords = coords[indices]
        point_labels = point_labels[indices]
    
    # Normalize coordinates to [-1, 1]
    coords = coords.astype(np.float32)
    for i in range(3):
        coords[:, i] = 2 * (coords[:, i] / segmentation.shape[i]) - 1
    
    return coords, point_labels


def process_patient(
    patient_dir,
    output_dir,
    target_spacing,
    target_size,
    normalize,
    extract_points,
    num_points,
):
    """
    Process a single patient.
    
    Args:
        patient_dir: Patient directory
        output_dir: Output directory
        target_spacing: Target voxel spacing
        target_size: Target image size
        normalize: Whether to normalize
        extract_points: Whether to extract point clouds
        num_points: Number of points to sample
    
    Returns:
        success: Whether processing was successful
    """
    patient_id = patient_dir.name
    output_patient_dir = output_dir / patient_id
    output_patient_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all frames
    image_files = sorted([f for f in patient_dir.glob('*_frame*.nii.gz') if '_gt' not in f.name])
    
    if len(image_files) == 0:
        print(f"Warning: No frames found for {patient_id}")
        return False
    
    # Load Info.cfg if exists
    info_file = patient_dir / 'Info.cfg'
    if info_file.exists():
        import shutil
        shutil.copy(info_file, output_patient_dir / 'Info.cfg')
    
    # Process each frame
    point_clouds = []
    
    for image_file in image_files:
        frame_name = image_file.stem.replace('.nii', '')
        
        # Load image
        image, affine, header = load_nifti(image_file)
        
        # Get original spacing
        original_spacing = header['pixdim'][1:4]
        
        # Resample
        if not np.allclose(original_spacing, target_spacing):
            image = resample_image(image, original_spacing, target_spacing, is_label=False)
        
        # Normalize
        if normalize:
            image = normalize_intensity(image)
        
        # Crop/pad
        image = crop_or_pad(image, target_size)
        
        # Save image
        output_image_file = output_patient_dir / f"{frame_name}.nii.gz"
        save_nifti(image, affine, header, output_image_file)
        
        # Process segmentation
        seg_file = image_file.parent / f"{frame_name}_gt.nii.gz"
        if seg_file.exists():
            seg, seg_affine, seg_header = load_nifti(seg_file)
            
            # Resample
            if not np.allclose(original_spacing, target_spacing):
                seg = resample_image(seg, original_spacing, target_spacing, is_label=True)
            
            # Crop/pad
            seg = crop_or_pad(seg, target_size)
            
            # Save segmentation
            output_seg_file = output_patient_dir / f"{frame_name}_gt.nii.gz"
            save_nifti(seg, seg_affine, seg_header, output_seg_file)
            
            # Extract point cloud
            if extract_points:
                points, labels = extract_point_cloud(seg, num_points)
                point_clouds.append({
                    'frame': frame_name,
                    'points': points.tolist(),
                    'labels': labels.tolist(),
                })
    
    # Save point clouds
    if extract_points and len(point_clouds) > 0:
        point_cloud_file = output_patient_dir / 'point_clouds.json'
        with open(point_cloud_file, 'w') as f:
            json.dump(point_clouds, f, indent=2)
    
    return True


def main():
    """Main function."""
    args = parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all patients
    patient_dirs = sorted([d for d in input_dir.glob('patient*') if d.is_dir()])
    
    print(f"Found {len(patient_dirs)} patients")
    print(f"Target spacing: {args.target_spacing}")
    print(f"Target size: {args.target_size}")
    print(f"Normalize: {args.normalize}")
    print(f"Extract points: {args.extract_points}")
    
    # Process each patient
    success_count = 0
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        try:
            success = process_patient(
                patient_dir,
                output_dir,
                args.target_spacing,
                args.target_size,
                args.normalize,
                args.extract_points,
                args.num_points,
            )
            if success:
                success_count += 1
        except Exception as e:
            print(f"Error processing {patient_dir.name}: {e}")
    
    print(f"\nPreprocessing completed!")
    print(f"Successfully processed: {success_count}/{len(patient_dirs)} patients")
    print(f"Output directory: {output_dir}")
    
    # Save preprocessing metadata
    metadata = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'target_spacing': args.target_spacing,
        'target_size': args.target_size,
        'normalize': args.normalize,
        'extract_points': args.extract_points,
        'num_points': args.num_points,
        'num_patients': len(patient_dirs),
        'success_count': success_count,
    }
    
    metadata_file = output_dir / 'preprocessing_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file}")


if __name__ == '__main__':
    main()
