import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List


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


class BasicBlock(nn.Module):
    """Basic block for ResNet with CBAM"""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, 
                 use_cbam=True, drop_path_rate=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_cbam = use_cbam
        
        if use_cbam:
            self.cbam = CBAM(planes)
            
        # Stochastic depth
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.use_cbam:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.drop_path(out) + identity
        out = self.relu(out)

        return out


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


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
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


class ResNet50CBAM(nn.Module):
    """ResNet-50 with CBAM attention for bird call classification"""
    
    def __init__(self, num_classes=30, in_channels=1, use_cbam=True, 
                 drop_path_rate=0.2, use_freq_attention=True):
        super(ResNet50CBAM, self).__init__()
        self.in_planes = 64
        self.use_cbam = use_cbam
        self.use_freq_attention = use_freq_attention
        
        # Modified first conv layer for single-channel spectrograms
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

    def forward(self, x, return_features=False):
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
        
        if return_features:
            return features
            
        x = self.fc(features)
        return x

    def get_attention_maps(self, x):
        """Extract attention maps for visualization"""
        attention_maps = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                if hasattr(module, 'cbam'):
                    attention_maps[name] = {
                        'channel': module.cbam.channel_attention(input[0]),
                        'spatial': module.cbam.spatial_attention(input[0])
                    }
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.named_modules():
            if isinstance(module, (BasicBlock, Bottleneck)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
            
        return attention_maps


def create_model(num_classes=30, pretrained=False, **kwargs):
    """Create ResNet-50 with CBAM model"""
    model = ResNet50CBAM(num_classes=num_classes, **kwargs)
    
    if pretrained:
        # Load pretrained weights if available
        # Note: You'll need to adapt ImageNet weights for single-channel input
        print("Loading pretrained weights not implemented for single-channel input")
    
    return model