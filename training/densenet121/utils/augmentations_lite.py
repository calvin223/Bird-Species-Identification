"""
Lightweight augmentation utilities optimized for limited VRAM.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random


class SpecAugmentLite:
    """Memory-efficient SpecAugment implementation."""
    
    def __init__(self, 
                 time_masks: int = 2,
                 time_mask_max: int = 30,
                 freq_masks: int = 2,
                 freq_mask_max: int = 20):
        self.time_masks = time_masks
        self.time_mask_max = time_mask_max
        self.freq_masks = freq_masks
        self.freq_mask_max = freq_mask_max
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to a spectrogram.
        Args:
            spectrogram: Tensor of shape (C, F, T) where C=1, F=128, T=259
        Returns:
            Augmented spectrogram
        """
        # Work with the same tensor to save memory
        spec = spectrogram.clone()
        _, freq_bins, time_steps = spec.shape
        
        # Time masking
        for _ in range(self.time_masks):
            t = random.randint(0, min(self.time_mask_max, time_steps - 1))
            t0 = random.randint(0, time_steps - t)
            spec[:, :, t0:t0+t] = 0
        
        # Frequency masking
        for _ in range(self.freq_masks):
            f = random.randint(0, min(self.freq_mask_max, freq_bins - 1))
            f0 = random.randint(0, freq_bins - f)
            spec[:, f0:f0+f, :] = 0
        
        return spec


class MixUpLite:
    """Memory-efficient MixUp implementation."""
    
    def __init__(self, alpha: float = 0.2, probability: float = 0.3):
        self.alpha = alpha
        self.probability = probability
    
    def __call__(self, 
                 x: torch.Tensor, 
                 y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp augmentation.
        Args:
            x: Input batch tensor
            y: Target labels
        Returns:
            Mixed inputs, target_a, target_b, lambda value
        """
        if random.random() > self.probability:
            return x, y, y, 1.0
        
        batch_size = x.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix inputs (in-place operation to save memory)
        mixed_x = lam * x + (1 - lam) * x[index]
        
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class TimeShift:
    """Time shifting augmentation."""
    
    def __init__(self, max_shift: int = 20, probability: float = 0.5):
        self.max_shift = max_shift
        self.probability = probability
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply time shifting to spectrogram.
        Args:
            spectrogram: Tensor of shape (C, F, T)
        Returns:
            Shifted spectrogram
        """
        if random.random() > self.probability:
            return spectrogram
        
        shift = random.randint(-self.max_shift, self.max_shift)
        
        if shift == 0:
            return spectrogram
        
        # Circular shift along time axis
        return torch.roll(spectrogram, shifts=shift, dims=-1)


class ComposeTransforms:
    """Compose multiple transforms with memory efficiency."""
    
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for transform in self.transforms:
            x = transform(x)
        return x


def get_training_augmentations(config: dict):
    """Get augmentation pipeline based on config."""
    transforms = []
    
    if config['augmentation']['time_shift']['enabled']:
        transforms.append(
            TimeShift(
                max_shift=config['augmentation']['time_shift']['max_shift'],
                probability=config['augmentation']['time_shift']['probability']
            )
        )
    
    if config['augmentation']['specaugment']['enabled']:
        transforms.append(
            SpecAugmentLite(
                time_masks=config['augmentation']['specaugment']['time_masks'],
                time_mask_max=config['augmentation']['specaugment']['time_mask_max'],
                freq_masks=config['augmentation']['specaugment']['freq_masks'],
                freq_mask_max=config['augmentation']['specaugment']['freq_mask_max']
            )
        )
    
    return ComposeTransforms(transforms)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """MixUp loss calculation."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class RandomErasing:
    """Random erasing augmentation (lightweight alternative to CutMix)."""
    
    def __init__(self, 
                 probability: float = 0.5,
                 area_ratio_range: Tuple[float, float] = (0.02, 0.33),
                 aspect_ratio_range: Tuple[float, float] = (0.3, 3.3)):
        self.probability = probability
        self.area_ratio_range = area_ratio_range
        self.aspect_ratio_range = aspect_ratio_range
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return img
        
        img = img.clone()
        _, h, w = img.shape
        area = h * w
        
        for _ in range(10):  # Try 10 times
            target_area = random.uniform(*self.area_ratio_range) * area
            aspect_ratio = random.uniform(*self.aspect_ratio_range)
            
            h_crop = int(round(np.sqrt(target_area * aspect_ratio)))
            w_crop = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w_crop < w and h_crop < h:
                x1 = random.randint(0, h - h_crop)
                y1 = random.randint(0, w - w_crop)
                img[:, x1:x1+h_crop, y1:y1+w_crop] = 0
                break
        
        return img


def normalize_batch(batch: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize batch to zero mean and unit variance.
    Memory efficient in-place operation.
    """
    # Calculate mean and std for each sample in the batch
    batch_size = batch.shape[0]
    for i in range(batch_size):
        sample = batch[i]
        mean = sample.mean()
        std = sample.std() + eps
        batch[i] = (sample - mean) / std
    
    return batch
