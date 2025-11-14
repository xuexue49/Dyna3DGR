"""
ACDC Dataset Loader

This module implements the data loader for the ACDC (Automated Cardiac Diagnosis Challenge) dataset.
The ACDC dataset contains cardiac MRI sequences with annotations for cardiac structure segmentation.

Dataset structure:
    data/ACDC/
    ├── patient001/
    │   ├── patient001_frame01.nii.gz
    │   ├── patient001_frame01_gt.nii.gz
    │   ├── patient001_frame02.nii.gz
    │   ├── patient001_frame02_gt.nii.gz
    │   ├── ...
    │   └── Info.cfg
    ├── patient002/
    └── ...
"""

import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import nibabel as nib
from pathlib import Path


class ACDCDataset(Dataset):
    """
    ACDC dataset for cardiac MRI sequences.
    
    The dataset loads 4D cardiac MRI sequences (3D + time) along with
    segmentation masks and metadata.
    """
    
    # Cardiac structure labels
    LABELS = {
        0: 'background',
        1: 'right_ventricle',
        2: 'myocardium',
        3: 'left_ventricle',
    }
    
    # Cardiac pathology groups
    PATHOLOGY = {
        'NOR': 'Normal',
        'MINF': 'Myocardial infarction',
        'DCM': 'Dilated cardiomyopathy',
        'HCM': 'Hypertrophic cardiomyopathy',
        'RV': 'Abnormal right ventricle',
    }
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (256, 256),
        num_frames: Optional[int] = None,
        normalize: bool = True,
        augmentation: bool = False,
        load_segmentation: bool = True,
        cache_data: bool = False,
    ):
        """
        Initialize ACDC dataset.
        
        Args:
            data_root: Root directory of ACDC dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (H, W)
            num_frames: Number of frames to sample (None = all frames)
            normalize: Whether to normalize images
            augmentation: Whether to apply data augmentation
            load_segmentation: Whether to load segmentation masks
            cache_data: Whether to cache loaded data in memory
        """
        super().__init__()
        
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_frames = num_frames
        self.normalize = normalize
        self.augmentation = augmentation
        self.load_segmentation = load_segmentation
        self.cache_data = cache_data
        
        # Find all patient directories
        self.patient_dirs = self._find_patients()
        
        # Split dataset
        self.patient_dirs = self._split_dataset()
        
        # Cache
        self.cache = {} if cache_data else None
        
        print(f"ACDC Dataset ({split}): {len(self.patient_dirs)} patients")
    
    def _find_patients(self) -> List[Path]:
        """Find all patient directories."""
        patient_dirs = []
        
        for patient_dir in sorted(self.data_root.glob('patient*')):
            if patient_dir.is_dir():
                # Check if directory contains data
                nii_files = list(patient_dir.glob('*.nii.gz'))
                if len(nii_files) > 0:
                    patient_dirs.append(patient_dir)
        
        return patient_dirs
    
    def _split_dataset(self) -> List[Path]:
        """
        Split dataset into train/val/test.
        
        Default split: 70% train, 15% val, 15% test
        """
        num_patients = len(self.patient_dirs)
        
        # Deterministic split based on patient number
        train_end = int(0.7 * num_patients)
        val_end = int(0.85 * num_patients)
        
        if self.split == 'train':
            return self.patient_dirs[:train_end]
        elif self.split == 'val':
            return self.patient_dirs[train_end:val_end]
        elif self.split == 'test':
            return self.patient_dirs[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _parse_patient_id(self, patient_dir: Path) -> str:
        """Extract patient ID from directory name."""
        return patient_dir.name
    
    def _load_info(self, patient_dir: Path) -> Dict:
        """
        Load patient information from Info.cfg file.
        
        Args:
            patient_dir: Patient directory
        
        Returns:
            info: Dictionary with patient information
        """
        info_file = patient_dir / 'Info.cfg'
        info = {}
        
        if info_file.exists():
            with open(info_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if ':' in line:
                        key, value = line.split(':', 1)
                        info[key.strip()] = value.strip()
        
        return info
    
    def _find_frames(self, patient_dir: Path) -> List[Tuple[Path, Path]]:
        """
        Find all frame files for a patient.
        
        Args:
            patient_dir: Patient directory
        
        Returns:
            frames: List of (image_path, segmentation_path) tuples
        """
        frames = []
        
        # Find all image files
        image_files = sorted(patient_dir.glob('*_frame*.nii.gz'))
        
        for image_file in image_files:
            # Skip segmentation files
            if '_gt' in image_file.name:
                continue
            
            # Find corresponding segmentation file
            seg_file = image_file.parent / image_file.name.replace('.nii.gz', '_gt.nii.gz')
            
            if self.load_segmentation and not seg_file.exists():
                continue
            
            frames.append((image_file, seg_file if self.load_segmentation else None))
        
        return frames
    
    def _load_nifti(self, file_path: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load NIfTI file.
        
        Args:
            file_path: Path to NIfTI file
        
        Returns:
            data: Image data array
            header: NIfTI header information
        """
        nii = nib.load(str(file_path))
        data = nii.get_fdata()
        
        # Extract header information
        header = {
            'affine': nii.affine,
            'pixdim': nii.header['pixdim'][1:4],  # Voxel spacing
            'dim': nii.header['dim'][1:4],  # Image dimensions
        }
        
        return data, header
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image.
        
        Args:
            image: Input image [H, W, D] or [H, W]
        
        Returns:
            processed: Preprocessed image
        """
        # Ensure 3D
        if image.ndim == 2:
            image = image[..., np.newaxis]
        
        # Normalize intensity
        if self.normalize:
            # Clip outliers
            p1, p99 = np.percentile(image, [1, 99])
            image = np.clip(image, p1, p99)
            
            # Normalize to [0, 1]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        return image
    
    def _resize_image(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int],
        is_label: bool = False,
    ) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image: Input image [H, W, D]
            target_size: Target size (H, W)
            is_label: Whether this is a label image (use nearest neighbor)
        
        Returns:
            resized: Resized image
        """
        from scipy.ndimage import zoom
        
        h, w = image.shape[:2]
        target_h, target_w = target_size
        
        # Calculate zoom factors
        zoom_factors = [target_h / h, target_w / w]
        
        # Add zoom factor for depth dimension if present
        if image.ndim == 3:
            zoom_factors.append(1.0)  # Don't resize depth
        
        # Resize
        order = 0 if is_label else 1  # Nearest neighbor for labels, linear for images
        resized = zoom(image, zoom_factors, order=order)
        
        return resized
    
    def _augment(
        self,
        image: np.ndarray,
        segmentation: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Apply data augmentation.
        
        Args:
            image: Input image
            segmentation: Segmentation mask (optional)
        
        Returns:
            augmented_image: Augmented image
            augmented_segmentation: Augmented segmentation
        """
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = np.flip(image, axis=1).copy()
            if segmentation is not None:
                segmentation = np.flip(segmentation, axis=1).copy()
        
        # Random rotation (small angle)
        if np.random.rand() > 0.5:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-10, 10)
            image = rotate(image, angle, axes=(0, 1), reshape=False, order=1)
            if segmentation is not None:
                segmentation = rotate(segmentation, angle, axes=(0, 1), reshape=False, order=0)
        
        # Random intensity shift
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            image = np.clip(image + shift, 0, 1)
        
        # Random intensity scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.9, 1.1)
            image = np.clip(image * scale, 0, 1)
        
        return image, segmentation
    
    def __len__(self) -> int:
        """Return number of patients in dataset."""
        return len(self.patient_dirs)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a patient's data.
        
        Args:
            idx: Patient index
        
        Returns:
            data: Dictionary containing:
                - images: [T, H, W, D] image sequence
                - segmentations: [T, H, W, D] segmentation sequence (if load_segmentation)
                - timestamps: [T] normalized time points
                - patient_id: Patient identifier
                - metadata: Additional metadata
        """
        # Check cache
        if self.cache is not None and idx in self.cache:
            return self.cache[idx]
        
        patient_dir = self.patient_dirs[idx]
        patient_id = self._parse_patient_id(patient_dir)
        
        # Load patient info
        info = self._load_info(patient_dir)
        
        # Find all frames
        frames = self._find_frames(patient_dir)
        
        if len(frames) == 0:
            raise ValueError(f"No frames found for patient {patient_id}")
        
        # Sample frames if needed
        if self.num_frames is not None and len(frames) > self.num_frames:
            # Uniform sampling
            indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            frames = [frames[i] for i in indices]
        
        # Load all frames
        images = []
        segmentations = [] if self.load_segmentation else None
        
        for image_path, seg_path in frames:
            # Load image
            image, header = self._load_nifti(image_path)
            image = self._preprocess_image(image)
            image = self._resize_image(image, self.image_size)
            images.append(image)
            
            # Load segmentation
            if self.load_segmentation and seg_path is not None:
                seg, _ = self._load_nifti(seg_path)
                seg = self._resize_image(seg, self.image_size, is_label=True)
                segmentations.append(seg)
        
        # Stack frames
        images = np.stack(images, axis=0)  # [T, H, W, D]
        if self.load_segmentation:
            segmentations = np.stack(segmentations, axis=0)  # [T, H, W, D]
        
        # Apply augmentation
        if self.augmentation and self.split == 'train':
            for t in range(images.shape[0]):
                images[t], seg_t = self._augment(
                    images[t],
                    segmentations[t] if self.load_segmentation else None
                )
                if self.load_segmentation:
                    segmentations[t] = seg_t
        
        # Create timestamps (normalized to [0, 1])
        timestamps = np.linspace(0, 1, len(frames))
        
        # Convert to tensors
        images = torch.from_numpy(images).float()
        timestamps = torch.from_numpy(timestamps).float()
        
        if self.load_segmentation:
            segmentations = torch.from_numpy(segmentations).long()
        
        # Prepare output
        data = {
            'images': images,
            'timestamps': timestamps,
            'patient_id': patient_id,
            'metadata': {
                'info': info,
                'num_frames': len(frames),
                'voxel_spacing': header['pixdim'].tolist(),
            }
        }
        
        if self.load_segmentation:
            data['segmentations'] = segmentations
        
        # Cache if enabled
        if self.cache is not None:
            self.cache[idx] = data
        
        return data


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for ACDC dataset.
    
    Handles variable sequence lengths by padding.
    
    Args:
        batch: List of data dictionaries
    
    Returns:
        batched: Batched data dictionary
    """
    # Find max sequence length
    max_length = max(item['images'].shape[0] for item in batch)
    
    # Prepare batched data
    batch_size = len(batch)
    image_shape = batch[0]['images'].shape[1:]
    
    images = torch.zeros(batch_size, max_length, *image_shape)
    timestamps = torch.zeros(batch_size, max_length)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    
    if 'segmentations' in batch[0]:
        segmentations = torch.zeros(batch_size, max_length, *image_shape, dtype=torch.long)
    else:
        segmentations = None
    
    patient_ids = []
    metadata_list = []
    
    for i, item in enumerate(batch):
        seq_len = item['images'].shape[0]
        
        images[i, :seq_len] = item['images']
        timestamps[i, :seq_len] = item['timestamps']
        lengths[i] = seq_len
        
        if segmentations is not None:
            segmentations[i, :seq_len] = item['segmentations']
        
        patient_ids.append(item['patient_id'])
        metadata_list.append(item['metadata'])
    
    batched = {
        'images': images,
        'timestamps': timestamps,
        'lengths': lengths,
        'patient_ids': patient_ids,
        'metadata': metadata_list,
    }
    
    if segmentations is not None:
        batched['segmentations'] = segmentations
    
    return batched


def create_acdc_dataloader(
    data_root: str,
    split: str = 'train',
    batch_size: int = 4,
    num_workers: int = 4,
    **dataset_kwargs
) -> torch.utils.data.DataLoader:
    """
    Create ACDC data loader.
    
    Args:
        data_root: Root directory of ACDC dataset
        split: Dataset split
        batch_size: Batch size
        num_workers: Number of worker processes
        **dataset_kwargs: Additional arguments for ACDCDataset
    
    Returns:
        dataloader: PyTorch DataLoader
    """
    dataset = ACDCDataset(data_root, split=split, **dataset_kwargs)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader
