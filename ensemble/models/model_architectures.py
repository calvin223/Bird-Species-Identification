"""
Model Loading Utilities for Ensemble
Handles loading and initialization of all 6 model architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import warnings
import numpy as np

warnings.filterwarnings('ignore')


# ============================================================================
# Model 1: EfficientNet-B1
# ============================================================================

class BirdCallEfficientNet(nn.Module):
    """EfficientNet-B1 model for bird call classification."""
    
    def __init__(self, num_classes: int = 30):
        super().__init__()
        try:
            from efficientnet_pytorch import EfficientNet
            self.backbone = EfficientNet.from_pretrained('efficientnet-b1')
            
            # Modify for single channel input
            conv_stem = self.backbone._conv_stem
            self.backbone._conv_stem = nn.Conv2d(
                1, conv_stem.out_channels,
                kernel_size=conv_stem.kernel_size,
                stride=conv_stem.stride,
                padding=conv_stem.padding,
                bias=False
            )
            
            num_features = self.backbone._fc.in_features
            self.backbone._fc = nn.Identity()
            
            self.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )
        except ImportError:
            raise ImportError("Please install efficientnet-pytorch: pip install efficientnet-pytorch")
    
    def forward(self, x):
        x = self.backbone(x)
        # Handle batch size of 1 for BatchNorm during eval
        if not self.training and x.size(0) == 1:
            # Use eval mode workaround for single sample
            x = self.classifier[0](x)  # Dropout
            x = self.classifier[1](x)  # Linear
            x = self.classifier[2](x)  # ReLU
            # Skip BatchNorm for single sample
            x = self.classifier[4](x)  # Dropout
            x = self.classifier[5](x)  # Linear
        else:
            x = self.classifier(x)
        return x


# ============================================================================
# Model 2: ResNet-50 with CBAM Attention (FIXED TO MATCH TRAINING)
# ============================================================================

class ChannelAttention(nn.Module):
    """Channel Attention module for CBAM"""
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention module for CBAM"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class FrequencyAttention(nn.Module):
    """Frequency-wise attention for audio spectrograms"""
    def __init__(self, in_channels, freq_bins=128):
        super(FrequencyAttention, self).__init__()
        self.freq_bins = freq_bins
        self.attention = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // 4, 1),
            nn.ReLU(),
            nn.Conv1d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: (B, C, F, T) where F=frequency, T=time
        B, C, freq_bins, T = x.size()
        # Global average pooling over time dimension
        freq_features = F.adaptive_avg_pool2d(x, (freq_bins, 1)).squeeze(-1)  # (B, C, freq_bins)
        freq_features = freq_features.transpose(1, 2)  # (B, freq_bins, C)
        
        # Apply frequency attention
        freq_att = self.attention(freq_features.transpose(1, 2))  # (B, C, F)
        freq_att = freq_att.unsqueeze(-1)  # (B, C, F, 1)
        
        return x * freq_att


class Bottleneck(nn.Module):
    """Bottleneck block for ResNet with CBAM"""
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, 
                 use_cbam=True, drop_path_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        
        if use_cbam:
            self.cbam = CBAM(self.expansion*planes)
            
        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.drop_path(out) + identity
        out = self.relu(out)

        return out


class ResNet50CBAM(nn.Module):
    """ResNet-50 with CBAM attention for bird call classification - EXACT TRAINING VERSION"""
    
    def __init__(self, num_classes: int = 30, in_channels: int = 1, use_cbam: bool = True, 
                 drop_path_rate: float = 0.2, use_freq_attention: bool = True):
        super(ResNet50CBAM, self).__init__()
        self.in_planes = 64
        self.use_cbam = use_cbam
        self.use_freq_attention = use_freq_attention
        
        # CRITICAL: Modified first conv layer with kernel_size=3 (NOT 7) as used in training!
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Calculate drop path rates for each block
        depths = [3, 4, 6, 3]  # ResNet-50 layer configuration
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # Build ResNet layers
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1, 
                                      drop_path_rates=dpr[0:3])
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2,
                                      drop_path_rates=dpr[3:7])
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2,
                                      drop_path_rates=dpr[7:13])
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2,
                                      drop_path_rates=dpr[13:16])
        
        # Frequency attention before global pooling
        if use_freq_attention:
            self.freq_attention = FrequencyAttention(512 * Bottleneck.expansion)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classification head
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, drop_path_rates):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample,
                           use_cbam=self.use_cbam, drop_path_rate=drop_path_rates[0]))
        self.in_planes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes, planes, use_cbam=self.use_cbam,
                               drop_path_rate=drop_path_rates[i]))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.use_freq_attention:
            x = self.freq_attention(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        x = self.fc(features)
        
        return x


# ============================================================================
# Model 3: DenseNet-121
# ============================================================================

class DenseNet121Custom(nn.Module):
    """Memory-efficient DenseNet-121 for bird call classification."""
    
    def __init__(self, num_classes: int = 30):
        super().__init__()
        from torchvision import models
        
        # Load pretrained DenseNet-121
        densenet = models.densenet121(weights='IMAGENET1K_V1')
        
        # Modify first conv layer for single channel
        self.features = densenet.features
        self.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add frequency attention
        self.freq_attention = nn.Sequential(
            nn.Conv1d(1024, 128, 1),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.Sigmoid()
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
    
    def forward(self, x):
        features = self.features(x)
        
        # Apply frequency attention
        b, c, h, w = features.size()
        freq_feat = features.mean(dim=3)
        freq_att = self.freq_attention(freq_feat)
        features = features * freq_att.unsqueeze(3)
        
        # Global average pooling
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        
        return out


# ============================================================================
# Model 4: Audio Spectrogram Transformer (AST)
# ============================================================================

class AudioSpectrogramTransformer(nn.Module):
    """Simplified AST for bird call classification."""
    
    def __init__(self, num_classes: int = 30, 
                 img_size: Tuple[int, int] = (128, 259),
                 patch_size: int = 16,
                 stride: int = 10,
                 embed_dim: int = 384,
                 depth: int = 6,
                 num_heads: int = 6,
                 mlp_ratio: float = 4.0,
                 drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        # Calculate number of patches
        self.grid_size = ((img_size[0] - patch_size) // stride + 1,
                         (img_size[1] - patch_size) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=stride)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, drop_rate, drop_path_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        
        # Add cls token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Classification head
        x = self.norm(x)
        x = x[:, 0]  # Use cls token
        x = self.head(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer block for AST."""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.1, drop_path=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + self.drop_path(attn_out)
        
        # MLP
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Drop path (stochastic depth) for regularization."""
    
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


# ============================================================================
# Model 5: ConvNeXt-Tiny (FIXED TO MATCH TRAINING)
# ============================================================================

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


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block - EXACT VERSION FROM TRAINING."""
    
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
    """A ConvNeXt stage."""
    
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
                x = torch.utils.checkpoint.checkpoint(block, x)
            else:
                x = block(x)
        
        return x


class ConvNeXtTiny(nn.Module):
    """ConvNeXt-Tiny for bird call classification - EXACT TRAINING VERSION."""
    
    def __init__(self, num_classes: int = 30,
                 in_channels: int = 1,
                 depths: list = [3, 3, 9, 3],
                 dims: list = [96, 192, 384, 768],
                 drop_path_rate: float = 0.1,
                 layer_scale_init_value: float = 1e-6,
                 stem_kernel_size: int = 4,
                 stem_stride: int = 4,
                 use_checkpoint: bool = True,
                 checkpoint_stages: list = [2, 3]):
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
        
        # Build stages
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
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Global average pooling
        x = x.mean([-2, -1])  # (B, C, H, W) -> (B, C)
        x = self.norm(x)
        x = self.head(x)
        
        return x


# ============================================================================
# Model 6: PANNs CNN14
# ============================================================================

class CNN14(nn.Module):
    """Simplified CNN14 architecture for bird call classification."""
    
    def __init__(self, num_classes: int = 30):
        super().__init__()
        
        # Convolutional layers
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_max_pool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(2048 * 2, num_classes)  # *2 for avg and max pool
    
    def forward(self, x):
        # Adapt input shape if needed
        if x.shape[-1] != 128:  # If not in PANNs format
            # Interpolate to match PANNs expected input
            x = F.interpolate(x, size=(1024, 128), mode='bilinear', align_corners=False)
        
        x = self.conv_block1(x)
        x = F.avg_pool2d(x, kernel_size=2)
        
        x = self.conv_block2(x)
        x = F.avg_pool2d(x, kernel_size=2)
        
        x = self.conv_block3(x)
        x = F.avg_pool2d(x, kernel_size=2)
        
        x = self.conv_block4(x)
        x = F.avg_pool2d(x, kernel_size=2)
        
        x = self.conv_block5(x)
        x = F.avg_pool2d(x, kernel_size=2)
        
        x = self.conv_block6(x)
        
        # Global pooling
        x_avg = self.global_avg_pool(x)
        x_max = self.global_max_pool(x)
        x = torch.cat([x_avg, x_max], dim=1)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ConvBlock(nn.Module):
    """Convolutional block for CNN14."""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


# ============================================================================
# Model Loading Functions
# ============================================================================

def load_model(model_type: str, model_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load a model based on its type and checkpoint path.
    
    Args:
        model_type: Type of model architecture
        model_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    # Create model based on type
    if model_type == 'efficientnet':
        model = BirdCallEfficientNet(num_classes=30)
    elif model_type == 'resnet50_cbam':
        model = ResNet50CBAM(num_classes=30)
    elif model_type == 'densenet121':
        model = DenseNet121Custom(num_classes=30)
    elif model_type == 'ast':
        model = AudioSpectrogramTransformer(num_classes=30)
    elif model_type == 'convnext':
        model = ConvNeXtTiny(num_classes=30)
    elif model_type == 'panns':
        model = CNN14(num_classes=30)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    try:
        # Use weights_only=False for compatibility with all checkpoint formats
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Load state dict
        model.load_state_dict(state_dict, strict=False)
        
    except Exception as e:
        print(f"Warning: Could not load checkpoint for {model_type}: {e}")
        print("Using random initialization")
    
    model.to(device)
    model.eval()
    
    return model


def get_model_info(model_type: str) -> Dict:
    """
    Get information about a model architecture.
    
    Args:
        model_type: Type of model
        
    Returns:
        Dictionary with model information
    """
    info = {
        'efficientnet': {
            'name': 'EfficientNet-B1',
            'params': '7.2M',
            'input_size': (1, 128, 259),
            'strengths': 'Excellent accuracy, well-calibrated confidence'
        },
        'resnet50_cbam': {
            'name': 'ResNet-50 with CBAM',
            'params': '28.2M',
            'input_size': (1, 128, 259),
            'strengths': 'Best test accuracy, attention mechanisms'
        },
        'densenet121': {
            'name': 'DenseNet-121',
            'params': '8.0M',
            'input_size': (1, 128, 259),
            'strengths': 'Memory efficient, dense connections'
        },
        'ast': {
            'name': 'Audio Spectrogram Transformer',
            'params': '10.9M',
            'input_size': (1, 128, 259),
            'strengths': 'Transformer architecture, temporal patterns'
        },
        'convnext': {
            'name': 'ConvNeXt-Tiny',
            'params': '27.8M',
            'input_size': (1, 128, 259),
            'strengths': 'Modern architecture, excellent performance'
        },
        'panns': {
            'name': 'PANNs CNN14',
            'params': '88.1M',
            'input_size': (1, 128, 259),
            'strengths': 'Pretrained on AudioSet, transfer learning'
        }
    }
    
    return info.get(model_type, {})
