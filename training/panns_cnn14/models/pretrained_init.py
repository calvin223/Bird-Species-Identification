"""
Initialize and load pretrained PANNs weights with proper adaptation.
"""

import os
import torch
import torch.nn as nn
from urllib.request import urlretrieve
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_pretrained_weights(url, save_path):
    """Download pretrained weights from URL."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if os.path.exists(save_path):
        print(f"Weights already exist at {save_path}")
        return save_path
        
    print(f"Downloading pretrained weights from {url}")
    
    with DownloadProgressBar(unit='B', unit_scale=True,
                           miniters=1, desc=os.path.basename(save_path)) as t:
        urlretrieve(url, filename=save_path, reporthook=t.update_to)
        
    return save_path


def create_pretrained_model(model_class, num_classes, checkpoint_url, 
                          checkpoint_path=None, device='cuda'):
    """Create model with pretrained weights."""
    
    # Download weights if needed
    if checkpoint_path is None:
        checkpoint_path = './pretrained/Cnn14_mAP=0.431.pth'
        
    if not os.path.exists(checkpoint_path):
        checkpoint_path = download_pretrained_weights(checkpoint_url, checkpoint_path)
        
    # Create model
    model = model_class(num_classes=num_classes)
    
    # Load pretrained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
        
    # Adapt state dict for our model
    adapted_state_dict = adapt_pretrained_weights(model, state_dict)
    
    # Load weights
    incompatible = model.load_state_dict(adapted_state_dict, strict=False)
    
    print(f"Loaded pretrained weights:")
    print(f"  Missing keys: {len(incompatible.missing_keys)}")
    print(f"  Unexpected keys: {len(incompatible.unexpected_keys)}")
    
    # Move to device
    model = model.to(device)
    
    return model


def adapt_pretrained_weights(model, pretrained_state_dict):
    """Adapt pretrained weights to match our model architecture."""
    
    # Get current model state dict
    model_state_dict = model.state_dict()
    
    # Create adapted state dict
    adapted_state_dict = {}
    
    # Map old keys to new keys if necessary
    key_mapping = {
        # Add any necessary key mappings here
        # 'old_key': 'new_key'
    }
    
    for key, value in pretrained_state_dict.items():
        # Skip incompatible layers
        if key in ['fc_audioset.weight', 'fc_audioset.bias']:
            continue
            
        # Skip spectrogram extraction layers (we use pre-computed spectrograms)
        if 'spectrogram_extractor' in key or 'logmel_extractor' in key:
            continue
            
        # Apply key mapping if exists
        if key in key_mapping:
            new_key = key_mapping[key]
        else:
            new_key = key
            
        # Check if key exists in model
        if new_key in model_state_dict:
            # Check shape compatibility
            if value.shape == model_state_dict[new_key].shape:
                adapted_state_dict[new_key] = value
            else:
                print(f"Shape mismatch for {new_key}: "
                      f"pretrained {value.shape} vs model {model_state_dict[new_key].shape}")
        else:
            # Try to find corresponding layer in CNN14 sub-module
            cnn14_key = f"cnn14.{new_key}"
            if cnn14_key in model_state_dict:
                if value.shape == model_state_dict[cnn14_key].shape:
                    adapted_state_dict[cnn14_key] = value
                    
    return adapted_state_dict


def initialize_model_stages(model, stage_config):
    """Initialize model for different training stages."""
    
    stage_name = stage_config['name']
    freeze_pattern = stage_config.get('freeze_pattern', None)
    
    print(f"Initializing stage: {stage_name}")
    
    # First unfreeze all
    for param in model.parameters():
        param.requires_grad = True
        
    # Then freeze based on pattern
    if freeze_pattern:
        import re
        frozen_params = 0
        total_params = 0
        
        # Simple pattern matching based on stage name
        for name, param in model.named_parameters():
            total_params += 1
            
            if stage_name == 'classifier_only':
                # Only train fc_audioset
                if 'fc_audioset' not in name:
                    param.requires_grad = False
                    frozen_params += 1
            elif stage_name == 'unfreeze_last_layers':
                # Train fc_audioset, fc1, and conv_block6
                if not any(x in name for x in ['fc_audioset', 'fc1', 'conv_block6']):
                    param.requires_grad = False
                    frozen_params += 1
            else:
                # Use standard regex matching
                if re.search(freeze_pattern, name):
                    param.requires_grad = False
                    frozen_params += 1
                
        print(f"  Frozen {frozen_params}/{total_params} parameters")
        
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"  Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model


def get_layer_groups(model):
    """Get parameter groups for different learning rates."""
    
    # Define layer groups
    groups = {
        'input_adapter': [],
        'early_conv': [],  # conv_block1-3
        'mid_conv': [],    # conv_block4-5
        'late_conv': [],   # conv_block6
        'classifier': [],  # fc layers
        'other': []
    }
    
    for name, param in model.named_parameters():
        if 'input_adapter' in name:
            groups['input_adapter'].append(param)
        elif any(f'conv_block{i}' in name for i in [1, 2, 3]):
            groups['early_conv'].append(param)
        elif any(f'conv_block{i}' in name for i in [4, 5]):
            groups['mid_conv'].append(param)
        elif 'conv_block6' in name:
            groups['late_conv'].append(param)
        elif 'fc' in name or 'classifier' in name:
            groups['classifier'].append(param)
        else:
            groups['other'].append(param)
            
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': groups['classifier'], 'lr_mult': 1.0},
        {'params': groups['late_conv'], 'lr_mult': 0.5},
        {'params': groups['mid_conv'], 'lr_mult': 0.3},
        {'params': groups['early_conv'], 'lr_mult': 0.1},
        {'params': groups['input_adapter'], 'lr_mult': 1.0},
        {'params': groups['other'], 'lr_mult': 0.5}
    ]
    
    # Remove empty groups
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    
    return param_groups


def reinitialize_classifier(model, num_classes, method='xavier'):
    """Reinitialize the classification layer."""
    
    if hasattr(model, 'cnn14'):
        classifier = model.cnn14.fc_audioset
    else:
        classifier = model.fc_audioset
        
    # Reinitialize weights
    if method == 'xavier':
        nn.init.xavier_uniform_(classifier.weight)
    elif method == 'kaiming':
        nn.init.kaiming_normal_(classifier.weight, mode='fan_out', nonlinearity='relu')
    elif method == 'normal':
        nn.init.normal_(classifier.weight, mean=0, std=0.01)
        
    # Initialize bias
    if classifier.bias is not None:
        nn.init.constant_(classifier.bias, 0)
        
    print(f"Reinitialized classifier with {method} initialization")


if __name__ == "__main__":
    # Test initialization
    from models.panns_cnn14 import BirdCallCNN14
    
    # URL for pretrained weights
    checkpoint_url = 'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
    
    # Create model with pretrained weights
    model = create_pretrained_model(
        BirdCallCNN14,
        num_classes=30,
        checkpoint_url=checkpoint_url,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
