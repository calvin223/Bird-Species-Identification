"""
ConvNeXt-Tiny model adapted for single-channel bird call spectrograms.
Optimized for memory efficiency on GTX 1650 (4GB VRAM).
Fixed version with corrected LayerNorm handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from functools import partial
from typing import List, Optional, Tuple
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors with 2D spatial dimensions."""
    
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)
    
    def forward(self, x):
        # x shape: (B, C, H, W)
        # Convert to (B, H, W, C) for layer norm
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # Convert back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block with memory optimizations."""
    
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6,
                 kernel_size=7, mlp_ratio=4):
        super().__init__()
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, 
                               padding=kernel_size//2, groups=dim)
        # Use standard LayerNorm after permute instead of LayerNorm2d
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        
        # Pointwise/1x1 convolutions
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, mlp_hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_hidden_dim, dim)
        
        # Layer scale
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                 requires_grad=True) if layer_scale_init_value > 0 else None
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x):
        input = x
        
        # Depthwise conv -> permute -> norm -> pointwise conv -> activation -> pointwise conv
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (B, C, H, W) -> (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        # Layer scale
        if self.gamma is not None:
            x = self.gamma * x
        
        x = x.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Residual connection with drop path
        x = input + self.drop_path(x)
        
        return x


class ConvNeXtStage(nn.Module):
    """A ConvNeXt stage with optional gradient checkpointing."""
    
    def __init__(self, in_channels, out_channels, depth, drop_path_rates,
                 layer_scale_init_value=1e-6, downsample=True,
                 use_checkpoint=False, kernel_size=7):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        
        # Downsampling layer
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_channels, eps=1e-6),
                nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            )
        else:
            self.downsample = nn.Identity()
        
        # ConvNeXt blocks
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                dim=out_channels,
                drop_path=drop_path_rates[i],
                layer_scale_init_value=layer_scale_init_value,
                kernel_size=kernel_size
            ) for i in range(depth)
        ])
    
    def forward(self, x):
        x = self.downsample(x)
        
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(block, x)
            else:
                x = block(x)
        
        return x


class ConvNeXtTinyBird(nn.Module):
    """ConvNeXt-Tiny adapted for bird call classification with memory optimizations."""
    
    def __init__(self, 
                 in_channels=1,
                 num_classes=30,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_path_rate=0.1,
                 layer_scale_init_value=1e-6,
                 stem_kernel_size=4,
                 stem_stride=4,
                 use_checkpoint=True,
                 checkpoint_stages=[2, 3]):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Stem - adapted for single channel spectrograms
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], 
                     kernel_size=stem_kernel_size, 
                     stride=stem_stride,
                     padding=stem_kernel_size//2),  # Add padding
            LayerNorm2d(dims[0], eps=1e-6)
        )
        
        # Stochastic depth schedule
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        
        # Build stages with memory optimization
        self.stages = nn.ModuleList()
        for i in range(4):
            use_checkpoint_stage = use_checkpoint and (i in checkpoint_stages)
            
            # Adjust kernel size for early stages (smaller for spectrograms)
            kernel_size = 5 if i < 2 else 7
            
            stage = ConvNeXtStage(
                in_channels=dims[i-1] if i > 0 else dims[0],
                out_channels=dims[i],
                depth=depths[i],
                drop_path_rates=drop_path_rates[cur:cur+depths[i]],
                layer_scale_init_value=layer_scale_init_value,
                downsample=(i > 0),  # No downsampling for first stage
                use_checkpoint=use_checkpoint_stage,
                kernel_size=kernel_size
            )
            self.stages.append(stage)
            cur += depths[i]
        
        # Global average pooling and classifier
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_features(self, x):
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.mean([-2, -1])  # (B, C, H, W) -> (B, C)
        x = self.norm(x)
        
        return x
    
    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.head = nn.Linear(self.head.in_features, num_classes)


def create_model(config):
    """Create ConvNeXt-Tiny model from config."""
    model = ConvNeXtTinyBird(
        in_channels=config.model.in_channels,
        num_classes=config.model.num_classes,
        depths=config.model.depths,
        dims=config.model.dims,
        drop_path_rate=config.model.drop_path_rate,
        layer_scale_init_value=config.model.layer_scale_init_value,
        stem_kernel_size=config.model.stem_kernel_size,
        stem_stride=config.model.stem_stride,
        use_checkpoint=config.model.use_checkpoint,
        checkpoint_stages=config.model.checkpoint_stages
    )
    
    return model


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_model():
    """Test the model with expected input shape."""
    model = ConvNeXtTinyBird()
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test with expected input shape
    x = torch.randn(2, 1, 128, 259)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Calculate model size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2
    print(f"Model size: {param_size:.2f} MB (params) + {buffer_size:.2f} MB (buffers)")


if __name__ == "__main__":
    test_model()
