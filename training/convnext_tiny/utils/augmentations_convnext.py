"""
Strong augmentations optimized for ConvNeXt on bird call spectrograms.
ConvNeXt benefits from aggressive augmentation due to its architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Union
import random


class SpecAugment(nn.Module):
    """SpecAugment augmentation for spectrograms."""
    
    def __init__(self, 
                 time_masks=2,
                 time_mask_param=30,
                 freq_masks=2, 
                 freq_mask_param=20,
                 p=1.0):
        super().__init__()
        self.time_masks = time_masks
        self.time_mask_param = time_mask_param
        self.freq_masks = freq_masks
        self.freq_mask_param = freq_mask_param
        self.p = p
    
    def forward(self, x, boost_factor=1.0):
        if not self.training or random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        
        # Adjust parameters with boost factor
        time_masks = int(self.time_masks * boost_factor)
        freq_masks = int(self.freq_masks * boost_factor)
        
        bs, c, freq, time = x.shape
        
        # Time masking
        for _ in range(time_masks):
            t = np.random.randint(0, min(self.time_mask_param, time))
            t0 = np.random.randint(0, time - t)
            x[:, :, :, t0:t0+t] = 0
        
        # Frequency masking
        for _ in range(freq_masks):
            f = np.random.randint(0, min(self.freq_mask_param, freq))
            f0 = np.random.randint(0, freq - f)
            x[:, :, f0:f0+f, :] = 0
        
        if squeeze_back:
            x = x.squeeze(0)
        
        return x


class MixUp(nn.Module):
    """MixUp augmentation with support for ConvNeXt."""
    
    def __init__(self, alpha=0.4, p=0.5):
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.enabled = True
    
    def forward(self, x, y=None):
        if not self.training or not self.enabled or random.random() > self.p:
            return x, y
        
        batch_size = x.size(0)
        indices = torch.randperm(batch_size, device=x.device)
        
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5
        
        mixed_x = lam * x + (1 - lam) * x[indices]
        
        if y is not None:
            y_a, y_b = y, y[indices]
            return mixed_x, (y_a, y_b, lam)
        
        return mixed_x, None
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        self.enabled = True


class CutMix(nn.Module):
    """CutMix augmentation - ConvNeXt handles this particularly well."""
    
    def __init__(self, alpha=1.0, p=0.3):
        super().__init__()
        self.alpha = alpha
        self.p = p
        self.enabled = True
    
    def forward(self, x, y=None):
        if not self.training or not self.enabled or random.random() > self.p:
            return x, y
        
        batch_size = x.size(0)
        indices = torch.randperm(batch_size, device=x.device)
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(x.size(), lam)
        
        # Apply CutMix
        x[:, :, bbx1:bbx2, bby1:bby2] = x[indices, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size(-1) * x.size(-2)))
        
        if y is not None:
            y_a, y_b = y, y[indices]
            return x, (y_a, y_b, lam)
        
        return x, None
    
    def _rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2
    
    def disable(self):
        self.enabled = False
    
    def enable(self):
        self.enabled = True


class RandomResizedCrop(nn.Module):
    """Random resized crop for spectrograms."""
    
    def __init__(self, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=1.0):
        super().__init__()
        self.scale = scale
        self.ratio = ratio
        self.p = p
    
    def forward(self, x, boost_factor=1.0):
        if not self.training or random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        
        # Adjust scale with boost factor
        scale_min = self.scale[0] / boost_factor
        scale = (max(0.5, scale_min), self.scale[1])
        
        bs, c, h, w = x.shape
        
        # Get random crop parameters
        area = h * w
        for _ in range(10):  # Max 10 attempts
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            new_w = int(round(np.sqrt(target_area * aspect_ratio)))
            new_h = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if new_w <= w and new_h <= h:
                i = random.randint(0, h - new_h)
                j = random.randint(0, w - new_w)
                
                # Crop
                x = x[:, :, i:i+new_h, j:j+new_w]
                
                # Resize back to original size
                x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
                
                if squeeze_back:
                    x = x.squeeze(0)
                
                return x
        
        # Fallback to original
        if squeeze_back:
            x = x.squeeze(0)
        return x


class GaussianNoise(nn.Module):
    """Add Gaussian noise to spectrograms."""
    
    def __init__(self, std=0.005, p=0.3):
        super().__init__()
        self.std = std
        self.p = p
    
    def forward(self, x, boost_factor=1.0):
        if not self.training or random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
        
        # Adjust noise with boost factor
        std = self.std * boost_factor
        
        noise = torch.randn_like(x) * std
        x = x + noise
        
        if squeeze_back:
            x = x.squeeze(0)
        
        return x


class TimeShift(nn.Module):
    """Shift spectrogram in time dimension."""
    
    def __init__(self, max_shift=20, p=0.5):
        super().__init__()
        self.max_shift = max_shift
        self.p = p
    
    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        # Roll along time dimension
        return torch.roll(x, shifts=shift, dims=-1)


class FrequencyShift(nn.Module):
    """Shift spectrogram in frequency dimension."""
    
    def __init__(self, max_shift=10, p=0.3):
        super().__init__()
        self.max_shift = max_shift
        self.p = p
    
    def forward(self, x):
        if not self.training or random.random() > self.p:
            return x
        
        shift = random.randint(-self.max_shift, self.max_shift)
        if shift == 0:
            return x
        
        # Roll along frequency dimension
        return torch.roll(x, shifts=shift, dims=-2)


class ConvNeXtAugmentation(nn.Module):
    """Complete augmentation pipeline for ConvNeXt."""
    
    def __init__(self, config):
        super().__init__()
        
        # Basic augmentations
        self.spec_augment = SpecAugment(
            time_masks=config.augmentation.spec_augment.time_masks,
            time_mask_param=config.augmentation.spec_augment.time_mask_param,
            freq_masks=config.augmentation.spec_augment.freq_masks,
            freq_mask_param=config.augmentation.spec_augment.freq_mask_param
        ) if config.augmentation.spec_augment.enabled else None
        
        self.random_crop = RandomResizedCrop(
            scale=config.augmentation.random_crop.scale,
            ratio=config.augmentation.random_crop.ratio
        ) if config.augmentation.random_crop.enabled else None
        
        self.gaussian_noise = GaussianNoise(
            std=config.augmentation.gaussian_noise.std,
            p=config.augmentation.gaussian_noise.prob
        ) if config.augmentation.gaussian_noise.enabled else None
        
        # Time and frequency shifts
        self.time_shift = TimeShift(max_shift=20, p=0.5)
        self.freq_shift = FrequencyShift(max_shift=10, p=0.3)
        
        # MixUp and CutMix are handled separately in training
        self.mixup = MixUp(
            alpha=config.augmentation.mixup.alpha,
            p=config.augmentation.mixup.prob
        ) if config.augmentation.mixup.enabled else None
        
        self.cutmix = CutMix(
            alpha=config.augmentation.cutmix.alpha,
            p=config.augmentation.cutmix.prob
        ) if config.augmentation.cutmix.enabled else None
    
    def forward(self, x, boost_factor=1.0):
        # Apply augmentations in sequence
        if self.random_crop is not None:
            x = self.random_crop(x, boost_factor)
        
        if self.spec_augment is not None:
            x = self.spec_augment(x, boost_factor)
        
        if self.gaussian_noise is not None:
            x = self.gaussian_noise(x, boost_factor)
        
        # Time and frequency shifts
        x = self.time_shift(x)
        x = self.freq_shift(x)
        
        return x
    
    def disable_mix_augmentations(self):
        """Disable MixUp and CutMix for validation."""
        if self.mixup is not None:
            self.mixup.disable()
        if self.cutmix is not None:
            self.cutmix.disable()
    
    def enable_mix_augmentations(self):
        """Enable MixUp and CutMix for training."""
        if self.mixup is not None:
            self.mixup.enable()
        if self.cutmix is not None:
            self.cutmix.enable()


def create_augmentation_pipeline(config):
    """Create augmentation pipeline from config."""
    return ConvNeXtAugmentation(config)


class TestTimeAugmentation:
    """Test-time augmentation for better predictions."""
    
    def __init__(self, n_augmentations=5):
        self.n_augmentations = n_augmentations
        
        # Define TTA transforms
        self.transforms = [
            lambda x: x,  # Original
            lambda x: torch.roll(x, shifts=10, dims=-1),   # Time shift right
            lambda x: torch.roll(x, shifts=-10, dims=-1),  # Time shift left
            lambda x: torch.roll(x, shifts=5, dims=-2),    # Freq shift up
            lambda x: torch.roll(x, shifts=-5, dims=-2),   # Freq shift down
        ]
    
    def __call__(self, x):
        """Apply TTA and return all augmented versions."""
        augmented = []
        for i in range(min(self.n_augmentations, len(self.transforms))):
            augmented.append(self.transforms[i](x))
        return torch.stack(augmented, dim=0)


if __name__ == "__main__":
    # Test augmentations
    x = torch.randn(2, 1, 128, 259)
    print(f"Original shape: {x.shape}")
    
    # Test individual augmentations
    spec_aug = SpecAugment()
    x_aug = spec_aug(x)
    print(f"After SpecAugment: {x_aug.shape}")
    
    mixup = MixUp()
    x_mix, y_mix = mixup(x, torch.tensor([0, 1]))
    print(f"After MixUp: {x_mix.shape}")
    
    cutmix = CutMix()
    x_cut, y_cut = cutmix(x, torch.tensor([0, 1]))
    print(f"After CutMix: {x_cut.shape}")
