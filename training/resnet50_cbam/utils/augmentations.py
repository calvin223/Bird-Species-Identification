import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class SpecAugment(nn.Module):
    """SpecAugment: A Simple Data Augmentation Method for ASR
    Reference: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(self, 
                 time_mask_param: int = 40,
                 freq_mask_param: int = 20,
                 n_time_masks: int = 2,
                 n_freq_masks: int = 2,
                 p: float = 1.0):
        """
        Args:
            time_mask_param: Maximum width of time mask
            freq_mask_param: Maximum width of frequency mask
            n_time_masks: Number of time masks to apply
            n_freq_masks: Number of frequency masks to apply
            p: Probability of applying augmentation
        """
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (channel, freq, time) or (batch, channel, freq, time)
        """
        if not self.training or np.random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        batch_size, channels, freq_bins, time_steps = x.shape
        
        # Clone to avoid modifying original
        x = x.clone()
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = min(int(np.random.uniform(0, self.time_mask_param)), time_steps - 1)
            if t > 0:
                t0 = np.random.randint(0, time_steps - t)
                x[:, :, :, t0:t0+t] = 0
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = min(int(np.random.uniform(0, self.freq_mask_param)), freq_bins - 1)
            if f > 0:
                f0 = np.random.randint(0, freq_bins - f)
                x[:, :, f0:f0+f, :] = 0
        
        return x if not squeeze_back else x.squeeze(0)


class TimeShift(nn.Module):
    """Random time shifting augmentation"""
    
    def __init__(self, shift_range: int = 20, p: float = 0.5):
        """
        Args:
            shift_range: Maximum shift in time steps
            p: Probability of applying augmentation
        """
        super().__init__()
        self.shift_range = shift_range
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (channel, freq, time) or (batch, channel, freq, time)
        """
        if not self.training or np.random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        batch_size, channels, freq_bins, time_steps = x.shape
        
        # Random shift for each sample in batch
        shifts = torch.randint(-self.shift_range, self.shift_range + 1, (batch_size,))
        
        # Apply shifts
        x_shifted = torch.zeros_like(x)
        for i, shift in enumerate(shifts):
            if shift == 0:
                x_shifted[i] = x[i]
            elif shift > 0:
                x_shifted[i, :, :, shift:] = x[i, :, :, :-shift]
            else:
                x_shifted[i, :, :, :shift] = x[i, :, :, -shift:]
        
        return x_shifted if not squeeze_back else x_shifted.squeeze(0)


class FrequencyMask(nn.Module):
    """Mask specific frequency bands (useful for bird calls)"""
    
    def __init__(self, 
                 mask_param: int = 10,
                 n_masks: int = 1,
                 replace_with_zero: bool = True,
                 p: float = 0.3):
        """
        Args:
            mask_param: Maximum width of frequency mask
            n_masks: Number of masks to apply
            replace_with_zero: If True, replace with zero; else with mean
            p: Probability of applying augmentation
        """
        super().__init__()
        self.mask_param = mask_param
        self.n_masks = n_masks
        self.replace_with_zero = replace_with_zero
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        batch_size, channels, freq_bins, time_steps = x.shape
        x = x.clone()
        
        for _ in range(self.n_masks):
            f = min(int(np.random.uniform(0, self.mask_param)), freq_bins - 1)
            if f > 0:
                f0 = np.random.randint(0, freq_bins - f)
                if self.replace_with_zero:
                    x[:, :, f0:f0+f, :] = 0
                else:
                    # Replace with mean value
                    mask_value = x[:, :, f0:f0+f, :].mean()
                    x[:, :, f0:f0+f, :] = mask_value
        
        return x if not squeeze_back else x.squeeze(0)


class CutMix(nn.Module):
    """CutMix augmentation for spectrograms"""
    
    def __init__(self, alpha: float = 1.0, p: float = 0.5):
        """
        Args:
            alpha: Beta distribution parameter
            p: Probability of applying CutMix
        """
        super().__init__()
        self.alpha = alpha
        self.p = p
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Args:
            x: Input tensor of shape (batch, channel, freq, time)
            labels: Labels tensor
        Returns:
            Mixed spectrograms, original labels, permuted labels, lambda value
        """
        if not self.training or np.random.random() > self.p:
            return x, labels, labels, 1.0
        
        batch_size = x.size(0)
        indices = torch.randperm(batch_size).to(x.device)
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Get random box
        _, _, H, W = x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_h = int(H * cut_rat)
        cut_w = int(W * cut_rat)
        
        # Center point
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        x = x.clone()
        x[:, :, bby1:bby2, bbx1:bbx2] = x[indices, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (H * W))
        
        return x, labels, labels[indices], lam


class GaussianNoise(nn.Module):
    """Add Gaussian noise to spectrograms"""
    
    def __init__(self, sigma: float = 0.05, p: float = 0.3):
        """
        Args:
            sigma: Standard deviation of Gaussian noise
            p: Probability of applying augmentation
        """
        super().__init__()
        self.sigma = sigma
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        noise = torch.randn_like(x) * self.sigma
        result = x + noise
        
        return result if not squeeze_back else result.squeeze(0)


class RandomAmplitude(nn.Module):
    """Random amplitude scaling"""
    
    def __init__(self, min_scale: float = 0.8, max_scale: float = 1.2, p: float = 0.5):
        """
        Args:
            min_scale: Minimum scaling factor
            max_scale: Maximum scaling factor
            p: Probability of applying augmentation
        """
        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or np.random.random() > self.p:
            return x
        
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze_back = True
        else:
            squeeze_back = False
            
        batch_size = x.size(0)
        scales = torch.FloatTensor(batch_size, 1, 1, 1).uniform_(self.min_scale, self.max_scale)
        scales = scales.to(x.device)
        
        result = x * scales
        
        return result if not squeeze_back else result.squeeze(0)


class ComposeAugmentations(nn.Module):
    """Compose multiple augmentations"""
    
    def __init__(self, augmentations):
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for aug in self.augmentations:
            x = aug(x)
        return x


def get_train_transforms(config: dict) -> ComposeAugmentations:
    """Get training augmentations from config"""
    augmentations = []
    
    if config.get('spec_augment', {}).get('enabled', True):
        spec_config = config['spec_augment']
        augmentations.append(SpecAugment(
            time_mask_param=spec_config.get('time_mask_param', 40),
            freq_mask_param=spec_config.get('freq_mask_param', 20),
            n_time_masks=spec_config.get('n_time_masks', 2),
            n_freq_masks=spec_config.get('n_freq_masks', 2),
            p=spec_config.get('p', 0.8)
        ))
    
    if config.get('time_shift', {}).get('enabled', True):
        time_config = config['time_shift']
        augmentations.append(TimeShift(
            shift_range=time_config.get('shift_range', 20),
            p=time_config.get('p', 0.5)
        ))
    
    if config.get('frequency_mask', {}).get('enabled', True):
        freq_config = config['frequency_mask']
        augmentations.append(FrequencyMask(
            mask_param=freq_config.get('mask_param', 10),
            n_masks=freq_config.get('n_masks', 1),
            p=freq_config.get('p', 0.3)
        ))
    
    if config.get('gaussian_noise', {}).get('enabled', True):
        noise_config = config['gaussian_noise']
        augmentations.append(GaussianNoise(
            sigma=noise_config.get('sigma', 0.05),
            p=noise_config.get('p', 0.3)
        ))
    
    if config.get('random_amplitude', {}).get('enabled', True):
        amp_config = config['random_amplitude']
        augmentations.append(RandomAmplitude(
            min_scale=amp_config.get('min_scale', 0.8),
            max_scale=amp_config.get('max_scale', 1.2),
            p=amp_config.get('p', 0.5)
        ))
    
    return ComposeAugmentations(augmentations)


class MixUpCriterion(nn.Module):
    """Loss function for MixUp training"""
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)


class CutMixCriterion(nn.Module):
    """Loss function for CutMix training"""
    
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
    
    def forward(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)