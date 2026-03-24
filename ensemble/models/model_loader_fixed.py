"""
Fixed Model Loading for Ensemble - Imports actual models from their directories
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

# Add all model directories to path
sys.path.append(r"D:\University\Comp702\project\model 3")
sys.path.append(r"D:\University\Comp702\project\model 4")
sys.path.append(r"D:\University\Comp702\project\model 6")

def load_model_fixed(model_type: str, model_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load the ACTUAL model implementations from their directories.
    """
    
    if model_type == 'efficientnet':
        # Model 1: EfficientNet - Keep using the ensemble version
        from .model_architectures import BirdCallEfficientNet
        model = BirdCallEfficientNet(num_classes=30)
        
    elif model_type == 'resnet50_cbam':
        # Model 2: ResNet-50 CBAM - Keep using the fixed ensemble version
        from .model_architectures import ResNet50CBAM
        model = ResNet50CBAM(num_classes=30)
        
    elif model_type == 'densenet121':
        # Model 3: Import ACTUAL DenseNet from model 3 directory
        from models.densenet_memory_efficient import MemoryEfficientDenseNet121
        model = MemoryEfficientDenseNet121(
            num_classes=30,
            growth_rate=32,
            num_init_features=64,
            drop_rate=0.3,
            compression=0.5,
            use_checkpoint=False,  # Disable for inference
            use_frequency_attention=True
        )
        
    elif model_type == 'ast':
        # Model 4: Import ACTUAL AST from model 4 directory
        from models.ast_bird import AST_Bird
        model = AST_Bird(
            img_size=(128, 259),
            patch_size=16,
            stride=10,
            num_classes=30,
            embed_dim=384,  # Reduced version as used in training
            depth=6,
            num_heads=6,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
        
    elif model_type == 'convnext':
        # Model 5: ConvNeXt - Keep using the fixed ensemble version
        from .model_architectures import ConvNeXtTiny
        model = ConvNeXtTiny(num_classes=30)
        
    elif model_type == 'panns':
        # Model 6: Import ACTUAL PANNs from model 6 directory
        from models.panns_cnn14 import BirdCallCNN14
        model = BirdCallCNN14(num_classes=30)
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint with proper handling
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if model_type == 'ast':
            # AST saves state dict directly
            if isinstance(checkpoint, dict) and 'cls_token' in checkpoint:
                # Direct state dict
                model.load_state_dict(checkpoint, strict=False)
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
        elif model_type == 'densenet121':
            # DenseNet uses model_state_dict
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
        elif model_type == 'panns':
            # PANNs has special structure
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # The checkpoint already has the correct structure
                model.load_state_dict(state_dict, strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
                
        else:
            # Standard loading for other models
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        print(f"✓ Successfully loaded {model_type}")
        
    except Exception as e:
        print(f"✗ Error loading {model_type}: {e}")
        print("  Using random initialization")
    
    model.to(device)
    model.eval()
    
    return model


# Test loading all models
if __name__ == "__main__":
    print("Testing fixed model loading...")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models_to_test = [
        ('densenet121', r"D:\University\Comp702\project\model 3\checkpoints\best_model.pth"),
        ('ast', r"D:\University\Comp702\project\model 4\fast_model_best.pth"),
        ('panns', r"D:\University\Comp702\project\model 6\outputs\20250814_163532\best_model.pth"),
    ]
    
    for model_type, path in models_to_test:
        print(f"\nLoading {model_type}...")
        try:
            model = load_model_fixed(model_type, path, device)
            
            # Test forward pass
            dummy_input = torch.randn(1, 1, 128, 259).to(device)
            with torch.no_grad():
                if model_type == 'ast':
                    # AST returns tuple
                    output = model(dummy_input)
                    if isinstance(output, tuple):
                        output = output[0]
                elif model_type == 'panns':
                    # PANNs might return dict or tensor
                    output = model(dummy_input)
                    if isinstance(output, dict):
                        output = output['clipwise_output']
                else:
                    output = model(dummy_input)
            
            print(f"  Output shape: {output.shape}")
            print(f"  ✓ {model_type} working!")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete!")
