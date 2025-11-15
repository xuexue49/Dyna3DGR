"""
Patient-specific data loader for per-case optimization.

This module provides data loading for single-patient training,
which is the standard approach for medical image reconstruction.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import nibabel as nib
from PIL import Image


class PatientDataset(Dataset):
    """
    Dataset for a single patient's 4D cardiac sequence.
    
    Loads all frames for one patient, maintaining temporal order.
    """
    
    def __init__(
        self,
        patient_dir: str,
        image_size: Tuple[int, int] = (256, 256),
        num_frames: Optional[int] = None,
        load_segmentation: bool = True,
        normalize: bool = True,
    ):
        """
        Initialize patient dataset.
        
        Args:
            patient_dir: Directory containing patient data
            image_size: Target image size (H, W) or (H, W, D)
            num_frames: Number of frames to sample (None = all)
            load_segmentation: Whether to load segmentation masks
            normalize: Whether to normalize intensities
        """
        self.patient_dir = Path(patient_dir)
        # Handle both 2D (H, W) and 3D (H, W, D) image_size
        if isinstance(image_size, (list, tuple)) and len(image_size) == 3:
            self.image_size = tuple(image_size[:2])  # Use only H, W for 2D resizing
            self.num_slices = image_size[2]
        else:
            self.image_size = tuple(image_size) if image_size else None
            self.num_slices = None
        self.load_segmentation = load_segmentation
        self.normalize = normalize
        
        # Load patient data
        self.frames = self._load_frames()
        
        # Sample frames if needed
        if num_frames and len(self.frames) > num_frames:
            indices = np.linspace(0, len(self.frames)-1, num_frames, dtype=int)
            self.frames = [self.frames[i] for i in indices]
        
        self.num_frames = len(self.frames)
        
        # Get patient metadata
        self.metadata = self._load_metadata()
    
    def _load_frames(self) -> List[Dict]:
        """
        Load all frames for this patient.
        
        Returns:
            frames: List of frame data dictionaries
        """
        frames = []
        
        # Check if data is in NIfTI format
        nifti_files = list(self.patient_dir.glob('*.nii*'))
        
        if nifti_files:
            # Load from NIfTI
            frames = self._load_from_nifti(nifti_files[0])
        else:
            # Load from PNG/JPG slices
            frames = self._load_from_images()
        
        return frames
    
    def _load_from_nifti(self, nifti_path: Path) -> List[Dict]:
        """Load frames from NIfTI file."""
        # Load NIfTI
        nifti = nib.load(str(nifti_path))
        data = nifti.get_fdata()
        
        # Get dimensions
        if data.ndim == 4:
            # 4D: (H, W, D, T)
            H, W, D, T = data.shape
            frames = []
            for t in range(T):
                frame_data = data[:, :, :, t]  # [H, W, D]
                frames.append({
                    'image': frame_data,
                    'spacing': nifti.header.get_zooms()[:3],
                })
        elif data.ndim == 3:
            # 3D: (H, W, T) - 2D+time
            H, W, T = data.shape
            frames = []
            for t in range(T):
                frame_data = data[:, :, t]  # [H, W]
                frames.append({
                    'image': frame_data,
                    'spacing': nifti.header.get_zooms()[:2],
                })
        else:
            raise ValueError(f"Unexpected NIfTI dimensions: {data.shape}")
        
        # Load segmentation if available
        if self.load_segmentation:
            seg_path = nifti_path.parent / nifti_path.name.replace('.nii', '_gt.nii')
            if seg_path.exists():
                seg_nifti = nib.load(str(seg_path))
                seg_data = seg_nifti.get_fdata()
                
                for t, frame in enumerate(frames):
                    if seg_data.ndim == 4:
                        frame['segmentation'] = seg_data[:, :, :, t]
                    elif seg_data.ndim == 3:
                        frame['segmentation'] = seg_data[:, :, t]
        
        return frames
    
    def _load_from_images(self) -> List[Dict]:
        """Load frames from image files."""
        # Look for images in subdirectories
        image_dir = self.patient_dir / 'images'
        if not image_dir.exists():
            image_dir = self.patient_dir
        
        # Get sorted image files
        image_files = sorted(image_dir.glob('*.png')) + sorted(image_dir.glob('*.jpg'))
        
        if len(image_files) == 0:
            raise ValueError(f"No image files found in {image_dir}")
        
        frames = []
        for img_path in image_files:
            # Load image
            img = Image.open(img_path).convert('L')  # Grayscale
            img_array = np.array(img, dtype=np.float32)
            
            frame_data = {'image': img_array}
            
            # Load segmentation if available
            if self.load_segmentation:
                seg_path = img_path.parent.parent / 'segmentations' / img_path.name
                if seg_path.exists():
                    seg = Image.open(seg_path).convert('L')
                    frame_data['segmentation'] = np.array(seg, dtype=np.float32)
            
            frames.append(frame_data)
        
        return frames
    
    def _load_metadata(self) -> Dict:
        """Load patient metadata."""
        metadata = {
            'patient_id': self.patient_dir.name,
            'num_frames': self.num_frames,
        }
        
        # Try to load Info.cfg if it exists (ACDC format)
        info_file = self.patient_dir / 'Info.cfg'
        if info_file.exists():
            with open(info_file, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return metadata
    
    def __len__(self) -> int:
        return self.num_frames
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single frame.
        
        Args:
            idx: Frame index
        
        Returns:
            sample: Dictionary containing:
                - image: Image tensor [H, W] or [H, W, D]
                - segmentation: Segmentation tensor (if available)
                - timestamp: Normalized timestamp [0, 1]
                - frame_idx: Frame index
        """
        frame = self.frames[idx]
        
        # Get image
        image = frame['image']
        
        # Resize if needed
        if image.ndim == 2:
            # 2D image
            image = self._resize_2d(image, self.image_size)
        elif image.ndim == 3:
            # 3D volume - resize each slice
            D = image.shape[2]
            resized_slices = []
            for d in range(D):
                slice_d = self._resize_2d(image[:, :, d], self.image_size)
                resized_slices.append(slice_d)
            image = np.stack(resized_slices, axis=2)
        
        # Normalize
        if self.normalize:
            image = self._normalize_intensity(image)
        
        # Convert to tensor
        image = torch.from_numpy(image).float()
        
        # Prepare output
        sample = {
            'image': image,
            'timestamp': idx / max(self.num_frames - 1, 1),
            'frame_idx': idx,
        }
        
        # Add segmentation if available
        if 'segmentation' in frame:
            seg = frame['segmentation']
            if seg.ndim == 2:
                seg = self._resize_2d(seg, self.image_size, is_segmentation=True)
            elif seg.ndim == 3:
                D = seg.shape[2]
                resized_slices = []
                for d in range(D):
                    slice_d = self._resize_2d(seg[:, :, d], self.image_size, is_segmentation=True)
                    resized_slices.append(slice_d)
                seg = np.stack(resized_slices, axis=2)
            
            sample['segmentation'] = torch.from_numpy(seg).float()
        
        return sample
    
    def _resize_2d(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        is_segmentation: bool = False,
    ) -> np.ndarray:
        """Resize 2D image."""
        from PIL import Image as PILImage
        
        if is_segmentation:
            # Use nearest neighbor for segmentation
            pil_img = PILImage.fromarray(image.astype(np.uint8))
            pil_img = pil_img.resize(target_size[::-1], PILImage.NEAREST)
        else:
            # Use bilinear for images
            pil_img = PILImage.fromarray(image)
            pil_img = pil_img.resize(target_size[::-1], PILImage.BILINEAR)
        
        return np.array(pil_img, dtype=np.float32)
    
    def _normalize_intensity(self, image: np.ndarray) -> np.ndarray:
        """Normalize image intensity to [0, 1]."""
        # Percentile clipping
        p_low, p_high = np.percentile(image[image > 0], [1, 99])
        image = np.clip(image, p_low, p_high)
        
        # Normalize to [0, 1]
        if p_high > p_low:
            image = (image - p_low) / (p_high - p_low)
        
        return image


def get_patient_ids(data_root: str, split: str = 'train') -> List[str]:
    """
    Get list of patient IDs for a given split.
    
    Args:
        data_root: Root directory containing patient folders
        split: 'train', 'val', or 'test'
    
    Returns:
        patient_ids: List of patient IDs
    """
    data_root = Path(data_root)
    
    # Get all patient directories
    all_patients = sorted([
        d.name for d in data_root.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])
    
    # Split dataset (70% train, 15% val, 15% test)
    n = len(all_patients)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    
    if split == 'train':
        return all_patients[:train_end]
    elif split == 'val':
        return all_patients[train_end:val_end]
    elif split == 'test':
        return all_patients[val_end:]
    else:
        raise ValueError(f"Unknown split: {split}")


def create_patient_dataloader(
    patient_dir: str,
    image_size: Tuple[int, int] = (256, 256),
    num_frames: Optional[int] = None,
    load_segmentation: bool = True,
    normalize: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create data loader for a single patient.
    
    Args:
        patient_dir: Directory containing patient data
        image_size: Target image size
        num_frames: Number of frames to sample
        load_segmentation: Whether to load segmentation
        normalize: Whether to normalize intensities
        num_workers: Number of worker processes
    
    Returns:
        dataloader: DataLoader for this patient
    """
    dataset = PatientDataset(
        patient_dir=patient_dir,
        image_size=image_size,
        num_frames=num_frames,
        load_segmentation=load_segmentation,
        normalize=normalize,
    )
    
    # Note: batch_size=1 and shuffle=False for per-patient training
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,  # Maintain temporal order
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataloader


def collate_patient_sequence(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for patient sequences.
    
    Since we process one patient at a time, this simply
    stacks frames along the time dimension.
    
    Args:
        batch: List of frame samples
    
    Returns:
        collated: Dictionary with batched tensors
    """
    # Stack frames
    images = torch.stack([item['image'] for item in batch], dim=0)
    timestamps = torch.tensor([item['timestamp'] for item in batch])
    frame_indices = torch.tensor([item['frame_idx'] for item in batch])
    
    collated = {
        'images': images,  # [T, H, W] or [T, H, W, D]
        'timestamps': timestamps,  # [T]
        'frame_indices': frame_indices,  # [T]
    }
    
    # Add segmentation if available
    if 'segmentation' in batch[0]:
        segmentations = torch.stack([item['segmentation'] for item in batch], dim=0)
        collated['segmentations'] = segmentations
    
    return collated
