"""
Memory-efficient DenseNet-121 implementation with gradient checkpointing and attention.
Optimized for 4GB VRAM limitation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from collections import OrderedDict
from typing import List, Tuple
import logging


class FrequencyAttention(nn.Module):
    """Attention module for frequency dimension of spectrograms."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, None))  # Pool over time, keep frequency
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, f, t = x.size()
        # Attention weights for each frequency bin
        y = self.avg_pool(x)  # Shape: (b, c, 1, t)
        y = y.mean(dim=3, keepdim=True)  # Shape: (b, c, 1, 1)
        y = self.fc(y)  # Shape: (b, c, 1, 1)
        return x * y.expand_as(x)


class _DenseLayer(nn.Module):
    """Single dense layer with optional memory checkpointing."""
    
    def __init__(self, 
                 num_input_features: int,
                 growth_rate: int,
                 bn_size: int,
                 drop_rate: float,
                 use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        # Bottleneck layers
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate,
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = drop_rate
    
    def forward(self, x):
        if self.use_checkpoint and x.requires_grad:
            # Use gradient checkpointing
            return checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        new_features = self.conv1(self.relu1(self.norm1(x)))
        new_features = self.conv2(self.relu2(self.norm2(new_features)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, 
                                   training=self.training)
        
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.ModuleDict):
    """Dense block with memory-efficient implementation."""
    
    def __init__(self,
                 num_layers: int,
                 num_input_features: int,
                 bn_size: int,
                 growth_rate: int,
                 drop_rate: float,
                 use_checkpoint: bool = True):
        super().__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                use_checkpoint=use_checkpoint
            )
            self.add_module(f'denselayer{i + 1}', layer)
    
    def forward(self, x):
        for name, layer in self.items():
            x = layer(x)
        return x


class _Transition(nn.Sequential):
    """Transition layer between dense blocks."""
    
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class MemoryEfficientDenseNet121(nn.Module):
    """
    DenseNet-121 optimized for bird call classification on limited VRAM.
    Features:
    - Gradient checkpointing on dense blocks
    - Frequency attention module
    - In-place operations where possible
    - Reduced memory footprint
    """
    
    def __init__(self,
                 num_classes: int = 30,
                 growth_rate: int = 32,
                 num_init_features: int = 64,
                 bn_size: int = 4,
                 drop_rate: float = 0.3,
                 compression: float = 0.5,
                 use_checkpoint: bool = True,
                 use_frequency_attention: bool = True):
        
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # First convolution - adapted for single channel spectrograms
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, 
                               stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))
        
        # Add frequency attention after initial features
        if use_frequency_attention:
            self.freq_attention = FrequencyAttention(num_init_features)
        else:
            self.freq_attention = None
        
        # DenseNet-121 configuration: [6, 12, 24, 16] layers in 4 blocks
        block_config = [6, 12, 24, 16]
        num_features = num_init_features
        
        # Build dense blocks
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                use_checkpoint=use_checkpoint
            )
            self.features.add_module(f'denseblock{i + 1}', block)
            num_features = num_features + num_layers * growth_rate
            
            # Add transition layer (except for last block)
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module(f'transition{i + 1}', trans)
                num_features = int(num_features * compression)
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))
        self.features.add_module('relu5', nn.ReLU(inplace=True))
        
        # Weighted average pooling for better feature aggregation
        self.weighted_pool = nn.Sequential(
            nn.Conv2d(num_features, num_features, 1),
            nn.Sigmoid()
        )
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Initialize weights
        self._initialize_weights()
        
        self.logger.info(
            f"Initialized MemoryEfficientDenseNet121: "
            f"classes={num_classes}, checkpoint={use_checkpoint}, "
            f"freq_attention={use_frequency_attention}"
        )
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with memory-efficient operations.
        Args:
            x: Input tensor of shape (B, 1, 128, 259)
        Returns:
            Output logits of shape (B, num_classes)
        """
        # Initial features
        features = self.features[:4](x)  # conv0, norm0, relu0, pool0
        
        # Apply frequency attention if enabled
        if self.freq_attention is not None:
            features = self.freq_attention(features)
        
        # Process through remaining features
        features = self.features[4:](features)
        
        # Weighted pooling
        weights = self.weighted_pool(features)
        features = features * weights
        
        # Global average pooling
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        
        # Classification
        out = self.classifier(out)
        
        return out
    
    def get_feature_extractor(self):
        """Get feature extractor (all layers except classifier)."""
        def extract_features(x):
            features = self.features(x)
            weights = self.weighted_pool(features)
            features = features * weights
            out = F.adaptive_avg_pool2d(features, (1, 1))
            return torch.flatten(out, 1)
        
        return extract_features


def create_densenet121(config: dict) -> nn.Module:
    """Create DenseNet-121 model from config."""
    model_config = config['model']
    
    model = MemoryEfficientDenseNet121(
        num_classes=config['dataset']['num_classes'],
        growth_rate=model_config['growth_rate'],
        num_init_features=model_config['num_init_features'],
        drop_rate=model_config['drop_rate'],
        compression=model_config['compression'],
        use_checkpoint=model_config['gradient_checkpointing'],
        use_frequency_attention=model_config['frequency_attention']
    )
    
    return model


def count_parameters(model: nn.Module) -> dict:
    """Count model parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params,
        'total_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
        'trainable_mb': trainable_params * 4 / (1024 * 1024)
    }
