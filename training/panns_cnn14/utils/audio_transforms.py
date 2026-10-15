"""
Audio transforms to convert our mel-spectrograms to PANNs format.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import Tuple, Optional, Dict, List


class SpectrogramAdapter(nn.Module):
    """Adapt mel-spectrograms between different formats."""
    
    def __init__(self, input_shape=(1, 128, 259), target_shape=(1, 1024, 128),
                 adaptation_method='interpolate'):
        super().__init__()
        self.input_shape = input_shape
        self.target_shape = target_shape
        self.adaptation_method = adaptation_method
        
        if adaptation_method == 'learned':
            # Learned adaptation with convolutional layers
            self.adapter = nn.Sequential(
                nn.Conv2d(input_shape[0], 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, target_shape[0], kernel_size=3, padding=1),
                nn.BatchNorm2d(target_shape[0])
            )
            
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channels, mel_bins, time_frames) or (channels, mel_bins, time_frames)
        Returns:
            Adapted tensor matching target shape
        """
        # Handle both batched and unbatched inputs
        if x.dim() == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
            
        if self.adaptation_method == 'interpolate':
            # Simple interpolation
            x = F.interpolate(x, size=self.target_shape[1:], 
                            mode='bilinear', align_corners=False)
                            
        elif self.adaptation_method == 'pad':
            # Pad to target size
            pad_time = self.target_shape[2] - x.shape[3]
            pad_freq = self.target_shape[1] - x.shape[2]
            
            if pad_time > 0:
                x = F.pad(x, (0, pad_time), mode='constant', value=0)
            else:
                x = x[:, :, :, :self.target_shape[2]]
                
            if pad_freq > 0:
                x = F.pad(x, (0, 0, 0, pad_freq), mode='constant', value=0)
            else:
                x = x[:, :, :self.target_shape[1], :]
                
        elif self.adaptation_method == 'learned':
            # Apply learned adaptation
            x = self.adapter(x)
            x = F.interpolate(x, size=self.target_shape[1:], 
                            mode='bilinear', align_corners=False)
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)
                            
        return x


class SpecAugment(nn.Module):
    """SpecAugment augmentation for spectrograms."""
    
    def __init__(self, time_masks=2, time_mask_param=64, 
                 freq_masks=2, freq_mask_param=16, prob=0.8):
        super().__init__()
        self.time_masks = time_masks
        self.time_mask_param = time_mask_param
        self.freq_masks = freq_masks
        self.freq_mask_param = freq_mask_param
        self.prob = prob
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (batch, channels, freq, time) or (channels, freq, time)
        """
        if random.random() > self.prob:
            return x
            
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            # Add batch dimension
            x = x.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
            
        batch_size, channels, freq_bins, time_frames = x.shape
        
        # Time masking
        for _ in range(self.time_masks):
            t = min(self.time_mask_param, time_frames)
            t = random.randint(0, t)
            if t > 0:
                t0 = random.randint(0, time_frames - t)
                x[:, :, :, t0:t0+t] = 0
                
        # Frequency masking
        for _ in range(self.freq_masks):
            f = min(self.freq_mask_param, freq_bins)
            f = random.randint(0, f)
            if f > 0:
                f0 = random.randint(0, freq_bins - f)
                x[:, :, f0:f0+f, :] = 0
                
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)
            
        return x


class MixUp(nn.Module):
    """MixUp augmentation for mel-spectrograms and labels."""
    
    def __init__(self, alpha=0.3, prob=0.8):
        super().__init__()
        self.alpha = alpha
        self.prob = prob
        
    def forward(self, x, y=None):
        """
        Args:
            x: Input spectrograms (batch, channels, freq, time) or (channels, freq, time)
            y: Labels (batch,) or one-hot (batch, num_classes)
        """
        if random.random() > self.prob:
            return x, y
            
        # Handle single sample input
        if x.dim() == 3:
            # Single sample, no mixing possible
            return x, y
            
        batch_size = x.shape[0]
        
        # Don't mix if batch size is 1
        if batch_size == 1:
            return x, y
        
        # Sample lambda from beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation for mixing
        index = torch.randperm(batch_size).to(x.device)
        
        # Mix spectrograms
        mixed_x = lam * x + (1 - lam) * x[index]
        
        # Mix labels if provided
        if y is not None:
            # Convert to one-hot if necessary
            if y.dim() == 1:
                num_classes = y.max().item() + 1
                y_onehot = torch.zeros(batch_size, num_classes).to(y.device)
                y_onehot.scatter_(1, y.unsqueeze(1), 1)
                y = y_onehot
                
            mixed_y = lam * y + (1 - lam) * y[index]
            return mixed_x, mixed_y
            
        return mixed_x, None


class RandomResizedCrop(nn.Module):
    """Random resized crop augmentation for spectrograms."""
    
    def __init__(self, freq_scale=(0.8, 1.0), time_scale=(0.8, 1.0), prob=0.5):
        super().__init__()
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.prob = prob
        
    def forward(self, x):
        if random.random() > self.prob:
            return x
            
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
            
        batch_size, channels, freq_bins, time_frames = x.shape
        
        # Random scale factors
        freq_factor = random.uniform(*self.freq_scale)
        time_factor = random.uniform(*self.time_scale)
        
        # New dimensions
        new_freq = int(freq_bins * freq_factor)
        new_time = int(time_frames * time_factor)
        
        # Random position
        freq_start = random.randint(0, freq_bins - new_freq)
        time_start = random.randint(0, time_frames - new_time)
        
        # Crop
        x_cropped = x[:, :, freq_start:freq_start+new_freq, 
                     time_start:time_start+new_time]
        
        # Resize back to original size
        x_resized = F.interpolate(x_cropped, size=(freq_bins, time_frames),
                                mode='bilinear', align_corners=False)
        
        # Remove batch dimension if input was unbatched
        if unbatched:
            x_resized = x_resized.squeeze(0)
            
        return x_resized


class TimeShift(nn.Module):
    """Time shift augmentation."""
    
    def __init__(self, shift_range=(-50, 50), prob=0.8):
        super().__init__()
        self.shift_range = shift_range
        self.prob = prob
        
    def forward(self, x):
        if random.random() > self.prob:
            return x
            
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
            
        shift = random.randint(*self.shift_range)
        
        if shift == 0:
            pass
        elif shift > 0:
            # Shift right
            x = F.pad(x[:, :, :, :-shift], (shift, 0), mode='constant', value=0)
        else:
            # Shift left
            x = F.pad(x[:, :, :, -shift:], (0, -shift), mode='constant', value=0)
            
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)
            
        return x


class GaussianNoise(nn.Module):
    """Add Gaussian noise to spectrograms."""
    
    def __init__(self, std=0.005, prob=0.5):
        super().__init__()
        self.std = std
        self.prob = prob
        
    def forward(self, x):
        if random.random() > self.prob:
            return x
            
        noise = torch.randn_like(x) * self.std
        return x + noise


class FrequencyMasking(nn.Module):
    """Frequency band masking augmentation."""
    
    def __init__(self, freq_masks=2, freq_width=10, prob=0.5):
        super().__init__()
        self.freq_masks = freq_masks
        self.freq_width = freq_width
        self.prob = prob
        
    def forward(self, x):
        if random.random() > self.prob:
            return x
            
        # Handle both 3D and 4D inputs
        if x.dim() == 3:
            x = x.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False
            
        _, _, freq_bins, _ = x.shape
        
        for _ in range(self.freq_masks):
            f = random.randint(0, self.freq_width)
            f0 = random.randint(0, freq_bins - f)
            x[:, :, f0:f0+f, :] = 0
            
        # Remove batch dimension if input was unbatched
        if unbatched:
            x = x.squeeze(0)
            
        return x


class ComposeTransforms(nn.Module):
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[nn.Module]):
        super().__init__()
        self.transforms = nn.ModuleList(transforms)
        
    def forward(self, x, y=None):
        for transform in self.transforms:
            if isinstance(transform, MixUp) and y is not None:
                x, y = transform(x, y)
            else:
                # For transforms that don't use labels, just pass x
                result = transform(x)
                # Handle transforms that might return tuples
                if isinstance(result, tuple):
                    x = result[0]
                else:
                    x = result
        return x, y


def create_train_transforms(config: Dict) -> ComposeTransforms:
    """Create training transforms from config."""
    
    transforms = []
    
    # Adapter to convert to PANNs format
    transforms.append(SpectrogramAdapter(
        input_shape=(1, 128, 259),
        target_shape=(1, 1024, 128),
        adaptation_method='interpolate'
    ))
    
    # Augmentations
    aug_config = config.get('augmentation', {})
    
    if aug_config.get('spec_augment', {}).get('prob', 0) > 0:
        sa_config = aug_config['spec_augment']
        transforms.append(SpecAugment(
            time_masks=int(sa_config['time_masks']),
            time_mask_param=int(sa_config['time_mask_param']),
            freq_masks=int(sa_config['freq_masks']),
            freq_mask_param=int(sa_config['freq_mask_param']),
            prob=float(sa_config['prob'])
        ))
        
    if aug_config.get('random_resized_crop', {}).get('prob', 0) > 0:
        rrc_config = aug_config['random_resized_crop']
        transforms.append(RandomResizedCrop(
            freq_scale=tuple(rrc_config['freq_scale']),
            time_scale=tuple(rrc_config['time_scale']),
            prob=float(rrc_config['prob'])
        ))
        
    if aug_config.get('time_shift', {}).get('prob', 0) > 0:
        ts_config = aug_config['time_shift']
        transforms.append(TimeShift(
            shift_range=tuple(ts_config['shift_range']),
            prob=float(ts_config['prob'])
        ))
        
    if aug_config.get('gaussian_noise', {}).get('prob', 0) > 0:
        gn_config = aug_config['gaussian_noise']
        transforms.append(GaussianNoise(
            std=float(gn_config['std']),
            prob=float(gn_config['prob'])
        ))
        
    if aug_config.get('frequency_masking', {}).get('prob', 0) > 0:
        fm_config = aug_config['frequency_masking']
        transforms.append(FrequencyMasking(
            freq_masks=int(fm_config['freq_masks']),
            freq_width=int(fm_config['freq_width']),
            prob=float(fm_config['prob'])
        ))
        
    return ComposeTransforms(transforms)


def create_val_transforms(config: Dict) -> nn.Module:
    """Create validation transforms (only adaptation, no augmentation)."""
    
    # Just use the adapter directly without ComposeTransforms
    return SpectrogramAdapter(
        input_shape=(1, 128, 259),
        target_shape=(1, 1024, 128),
        adaptation_method='interpolate'
    )


# Test-time augmentation transforms
def create_tta_transforms(num_augmentations: int = 5) -> List[nn.Module]:
    """Create test-time augmentation transforms."""
    
    tta_transforms = []
    
    # Original (no augmentation)
    tta_transforms.append(SpectrogramAdapter(
        input_shape=(1, 128, 259),
        target_shape=(1, 1024, 128),
        adaptation_method='interpolate'
    ))
    
    # Different time shifts
    for shift in [-30, -15, 15, 30]:
        transforms = [
            SpectrogramAdapter(
                input_shape=(1, 128, 259),
                target_shape=(1, 1024, 128),
                adaptation_method='interpolate'
            ),
            TimeShift(shift_range=(shift, shift), prob=1.0)
        ]
        tta_transforms.append(ComposeTransforms(transforms))
        
        if len(tta_transforms) >= num_augmentations:
            break
            
    return tta_transforms[:num_augmentations]


if __name__ == "__main__":
    # Test transforms
    config = {
        'augmentation': {
            'spec_augment': {
                'time_masks': 2,
                'time_mask_param': 64,
                'freq_masks': 2,
                'freq_mask_param': 16,
                'prob': 0.8
            },
            'mixup': {
                'alpha': 0.3,
                'prob': 0.8
            }
        }
    }
    
    # Create transforms
    train_transforms = create_train_transforms(config)
    val_transforms = create_val_transforms(config)
    
    # Test with random data
    x = torch.randn(4, 1, 128, 259)
    y = torch.randint(0, 30, (4,))
    
    # Apply transforms
    x_aug, y_aug = train_transforms(x, y)
    x_val, _ = val_transforms(x)
    
    print(f"Original shape: {x.shape}")
    print(f"Augmented shape: {x_aug.shape}")
    print(f"Validation shape: {x_val.shape}")
