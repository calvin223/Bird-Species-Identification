"""
Robust Model Loading with Direct File Imports
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import importlib.util

# Helper function to import module from file path
def import_module_from_file(module_name, file_path):
    """Import a module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Set up base path
base_path = Path(r"D:\University\Comp702\project")

# Add ensemble models path
sys.path.insert(0, str(base_path / "ensemble model" / "models"))

def load_model_fixed(model_type: str, model_path: str, device: str = 'cuda') -> nn.Module:
    """
    Load models with direct imports and robust checkpoint handling.
    """
    
    print(f"  Loading {model_type}...")
    
    try:
        if model_type == 'efficientnet':
            # Model 1: EfficientNet
            from model_architectures import BirdCallEfficientNet
            model = BirdCallEfficientNet(num_classes=30)
            
        elif model_type == 'resnet50_cbam':
            # Model 2: ResNet-50 CBAM
            from model_architectures import ResNet50CBAM
            model = ResNet50CBAM(num_classes=30)
            
        elif model_type == 'densenet121':
            # Model 3: DenseNet - Direct import
            densenet_module = import_module_from_file(
                "densenet_memory_efficient",
                base_path / "model 3" / "models" / "densenet_memory_efficient.py"
            )
            model = densenet_module.MemoryEfficientDenseNet121(
                num_classes=30,
                growth_rate=32,
                num_init_features=64,
                drop_rate=0.3,
                compression=0.5,
                use_checkpoint=False,
                use_frequency_attention=True
            )
            
        elif model_type == 'ast':
            # Model 4: AST - Direct import
            ast_module = import_module_from_file(
                "ast_bird",
                base_path / "model 4" / "models" / "ast_bird.py"
            )
            model = ast_module.AST_Bird(
                img_size=(128, 259),
                patch_size=16,
                stride=10,
                num_classes=30,
                embed_dim=384,
                depth=6,
                num_heads=6,
                drop_rate=0.1,
                drop_path_rate=0.1
            )
            
        elif model_type == 'convnext':
            # Model 5: ConvNeXt
            from model_architectures import ConvNeXtTiny
            model = ConvNeXtTiny(num_classes=30)
            
        elif model_type == 'panns':
            # Model 6: PANNs - Direct import
            panns_module = import_module_from_file(
                "panns_cnn14",
                base_path / "model 6" / "models" / "panns_cnn14.py"
            )
            model = panns_module.BirdCallCNN14(num_classes=30)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load checkpoint with proper handling
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint structures
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                # Raw state dict (shouldn't happen with fixed AST)
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load with strict=False to handle minor mismatches
        model.load_state_dict(state_dict, strict=False)
        print(f"    ✓ Loaded successfully")
        
    except Exception as e:
        print(f"    ✗ Error: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    return model
