"""
PANNs CNN14 architecture adapted for bird call classification.
Based on the original PANNs implementation with modifications for our specific use case.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
import warnings


def init_layer(layer):
    """Initialize a Linear or Convolutional layer."""
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer."""
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect pool_type!')
        
        return x


class AttentionPooling(nn.Module):
    """Attention-based pooling layer for better feature aggregation."""
    def __init__(self, embed_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 4, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, channels, time, freq)
        batch_size, channels, time, freq = x.shape
        
        # Reshape to (batch, channels, time*freq)
        x = x.view(batch_size, channels, -1)
        
        # Transpose to (batch, time*freq, channels)
        x = x.transpose(1, 2)
        
        # Calculate attention weights
        att_weights = self.attention(x)  # (batch, time*freq, 1)
        att_weights = F.softmax(att_weights, dim=1)
        
        # Apply attention
        x = torch.sum(x * att_weights, dim=1)  # (batch, channels)
        
        return x, att_weights


class Cnn14(nn.Module):
    def __init__(self, classes_num=30, feature_fusion=True, use_attention_pooling=True):
        super(Cnn14, self).__init__()
        
        self.feature_fusion = feature_fusion
        self.use_attention_pooling = use_attention_pooling
        
        # Input adaptation layer for our mel-spectrograms
        self.input_adapter = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)
        
        # CNN blocks
        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)
        
        # Additional embedding layers
        self.fc1 = nn.Linear(2048, 2048, bias=True)
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head
        self.fc_audioset = nn.Linear(2048, classes_num, bias=True)
        
        # Optional attention pooling
        if use_attention_pooling:
            self.attention_pooling = AttentionPooling(2048)
            
        # Feature fusion from multiple layers
        if feature_fusion:
            self.fusion_conv = nn.Conv2d(512 + 1024 + 2048, 2048, 
                                       kernel_size=1, stride=1, padding=0)
            init_layer(self.fusion_conv)
            
        self.init_weights()
        
    def init_weights(self):
        init_layer(self.fc1)
        init_layer(self.fc_audioset)
        init_layer(self.input_adapter)
        
    def forward(self, input, return_embedding=False):
        """
        Args:
            input: (batch_size, 1, time_frames, mel_bins)
            return_embedding: If True, return embeddings instead of logits
        """
        # Adapt input if necessary
        x = self.adapt_input(input)
        
        # Forward through CNN blocks
        x1 = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x2 = self.conv_block2(x1, pool_size=(2, 2), pool_type='avg')
        x3 = self.conv_block3(x2, pool_size=(2, 2), pool_type='avg')
        x4 = self.conv_block4(x3, pool_size=(2, 2), pool_type='avg')
        x5 = self.conv_block5(x4, pool_size=(2, 2), pool_type='avg')
        x6 = self.conv_block6(x5, pool_size=(1, 1), pool_type='avg')
        
        # Feature fusion from multiple scales
        if self.feature_fusion:
            # Resize feature maps to match dimensions
            h, w = x6.shape[2], x6.shape[3]
            x4_resized = F.adaptive_avg_pool2d(x4, (h, w))
            x5_resized = F.adaptive_avg_pool2d(x5, (h, w))
            
            # Concatenate and fuse
            x_concat = torch.cat([x4_resized, x5_resized, x6], dim=1)
            x = self.fusion_conv(x_concat)
        else:
            x = x6
            
        # Pooling
        if self.use_attention_pooling:
            x, att_weights = self.attention_pooling(x)
        else:
            x = torch.mean(x, dim=(2, 3))
            att_weights = None
            
        # Embedding
        x = F.relu(self.fc1(x))
        embedding = x
        x = self.dropout(x)
        
        if return_embedding:
            return embedding
            
        # Classification
        clipwise_output = self.fc_audioset(x)
        
        output_dict = {
            'clipwise_output': clipwise_output,
            'embedding': embedding,
            'attention_weights': att_weights
        }
        
        return output_dict
        
    def adapt_input(self, x):
        """Adapt input mel-spectrograms to expected format."""
        # Input shape: (batch, 1, 128, 259)
        # Target shape: (batch, 1, 1024, 128) for ~10 second clips
        
        batch_size = x.shape[0]
        
        # Option 1: Interpolate to match expected dimensions
        # This maintains the full spectral information
        x = F.interpolate(x, size=(1024, 128), mode='bilinear', align_corners=False)
        
        # Apply input adapter
        x = self.input_adapter(x)
        
        return x
        
    def extract_features(self, input):
        """Extract intermediate features for analysis."""
        features = {}
        
        x = self.adapt_input(input)
        
        x1 = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        features['conv1'] = x1
        
        x2 = self.conv_block2(x1, pool_size=(2, 2), pool_type='avg')
        features['conv2'] = x2
        
        x3 = self.conv_block3(x2, pool_size=(2, 2), pool_type='avg')
        features['conv3'] = x3
        
        x4 = self.conv_block4(x3, pool_size=(2, 2), pool_type='avg')
        features['conv4'] = x4
        
        x5 = self.conv_block5(x4, pool_size=(2, 2), pool_type='avg')
        features['conv5'] = x5
        
        x6 = self.conv_block6(x5, pool_size=(1, 1), pool_type='avg')
        features['conv6'] = x6
        
        return features


class BirdCallCNN14(nn.Module):
    """
    Wrapper around CNN14 with bird-specific adaptations.
    """
    def __init__(self, num_classes=30, pretrained_path=None, 
                 use_hierarchical=False, bird_groups=None):
        super(BirdCallCNN14, self).__init__()
        
        self.num_classes = num_classes
        self.use_hierarchical = use_hierarchical
        self.bird_groups = bird_groups
        
        # Base CNN14 model
        self.cnn14 = Cnn14(classes_num=num_classes, 
                          feature_fusion=True, 
                          use_attention_pooling=True)
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
            
        # Hierarchical classification head if enabled
        if use_hierarchical and bird_groups:
            self.group_classifier = nn.Linear(2048, len(bird_groups))
            init_layer(self.group_classifier)
            
    def forward(self, x, return_all=False):
        # Get base model outputs
        outputs = self.cnn14(x)
        
        if self.use_hierarchical:
            embedding = outputs['embedding']
            group_logits = self.group_classifier(embedding)
            outputs['group_output'] = group_logits
            
        if return_all:
            return outputs
        else:
            return outputs['clipwise_output']
            
    def load_pretrained_weights(self, checkpoint_path):
        """Load pretrained AudioSet weights."""
        print(f"Loading pretrained weights from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # Remove incompatible layers
        incompatible_keys = ['fc_audioset.weight', 'fc_audioset.bias', 
                           'spectrogram_extractor', 'logmel_extractor']
        
        for key in incompatible_keys:
            if key in state_dict:
                del state_dict[key]
                
        # Load weights
        missing_keys, unexpected_keys = self.cnn14.load_state_dict(
            state_dict, strict=False)
        
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # Initialize new classifier
        init_layer(self.cnn14.fc_audioset)
        
    def freeze_base(self, freeze_pattern=None):
        """Freeze layers based on pattern."""
        if freeze_pattern is None:
            # Freeze all conv blocks
            for name, param in self.cnn14.named_parameters():
                if 'conv_block' in name:
                    param.requires_grad = False
        else:
            # Freeze based on pattern
            import re
            for name, param in self.cnn14.named_parameters():
                if re.match(freeze_pattern, name):
                    param.requires_grad = False
                    
    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test the model
    model = BirdCallCNN14(num_classes=30)
    
    # Test input
    x = torch.randn(4, 1, 128, 259)
    
    # Forward pass
    outputs = model(x, return_all=True)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {outputs['clipwise_output'].shape}")
    print(f"Embedding shape: {outputs['embedding'].shape}")
