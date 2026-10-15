"""
Layer-wise learning rate decay for ConvNeXt.
Earlier layers learn slower than later layers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import json


def get_parameter_groups(model: nn.Module,
                        base_lr: float,
                        weight_decay: float,
                        layer_decay: float = 0.7) -> List[Dict]:
    """
    Create parameter groups with layer-wise learning rate decay.
    
    Args:
        model: ConvNeXt model
        base_lr: Base learning rate for the last layer
        weight_decay: Weight decay value
        layer_decay: Decay factor for earlier layers
        
    Returns:
        List of parameter groups for optimizer
    """
    
    # Get all parameters with their names
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    # Assign layer indices
    def get_layer_id(name):
        """Get layer ID for layer-wise lr decay."""
        if name.startswith('stem'):
            return 0
        elif name.startswith('stages.0'):
            return 1
        elif name.startswith('stages.1'):
            return 2
        elif name.startswith('stages.2'):
            return 3
        elif name.startswith('stages.3'):
            return 4
        elif name.startswith('norm') or name.startswith('head'):
            return 5
        else:
            return 5  # Default to last layer
    
    # Group parameters by layer
    layers = {}
    for pn, p in param_dict.items():
        layer_id = get_layer_id(pn)
        if layer_id not in layers:
            layers[layer_id] = []
        layers[layer_id].append((pn, p))
    
    # Create parameter groups with decayed learning rates
    parameter_groups = []
    max_layer = max(layers.keys())
    
    for layer_id in sorted(layers.keys()):
        # Calculate learning rate for this layer
        layer_lr = base_lr * (layer_decay ** (max_layer - layer_id))
        
        # Separate parameters that need weight decay
        decay_params = []
        no_decay_params = []
        
        for pn, p in layers[layer_id]:
            if not p.requires_grad:
                continue
                
            # No weight decay for biases and normalization parameters
            if pn.endswith('.bias') or 'norm' in pn or 'gamma' in pn:
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        
        # Add parameter groups
        if decay_params:
            parameter_groups.append({
                'params': decay_params,
                'lr': layer_lr,
                'weight_decay': weight_decay,
                'layer_id': layer_id,
                'param_names': [pn for pn, param in layers[layer_id] if any(param is p for p in decay_params)]
            })
        
        if no_decay_params:
            parameter_groups.append({
                'params': no_decay_params,
                'lr': layer_lr,
                'weight_decay': 0.0,
                'layer_id': layer_id,
                'param_names': [pn for pn, param in layers[layer_id] if any(param is p for p in no_decay_params)]
            })
    
    return parameter_groups


class LayerWiseLRScheduler:
    """Custom scheduler that maintains layer-wise LR ratios."""
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 base_scheduler,
                 layer_decay: float = 0.7):
        """
        Args:
            optimizer: Optimizer with layer-wise parameter groups
            base_scheduler: Base scheduler (e.g., CosineAnnealingLR)
            layer_decay: Decay factor between layers
        """
        self.optimizer = optimizer
        self.base_scheduler = base_scheduler
        self.layer_decay = layer_decay
        
        # Store initial LRs for each group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self, epoch=None):
        """Step the scheduler."""
        # Step base scheduler
        self.base_scheduler.step(epoch)
        
        # Get the current LR from base scheduler
        current_base_lr = self.base_scheduler.get_last_lr()[0]
        
        # Find the scaling factor
        if len(self.base_lrs) > 0:
            scale_factor = current_base_lr / self.base_lrs[-1]
        else:
            scale_factor = 1.0
        
        # Apply scaled LRs to all groups
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * scale_factor
    
    def get_last_lr(self) -> List[float]:
        """Get last learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def state_dict(self) -> Dict:
        """Get scheduler state."""
        return {
            'base_scheduler': self.base_scheduler.state_dict(),
            'base_lrs': self.base_lrs,
            'layer_decay': self.layer_decay
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Load scheduler state."""
        self.base_scheduler.load_state_dict(state_dict['base_scheduler'])
        self.base_lrs = state_dict['base_lrs']
        self.layer_decay = state_dict['layer_decay']


def create_optimizer_with_layer_decay(model: nn.Module,
                                    config) -> Tuple[torch.optim.Optimizer, Optional[LayerWiseLRScheduler]]:
    """
    Create optimizer with layer-wise learning rate decay.
    
    Args:
        model: ConvNeXt model
        config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    
    # Check if layer-wise decay is enabled
    if config.training.layer_lr_decay.enabled:
        # Get parameter groups with layer-wise decay
        param_groups = get_parameter_groups(
            model,
            base_lr=config.training.optimizer.lr,
            weight_decay=config.training.optimizer.weight_decay,
            layer_decay=config.training.layer_lr_decay.decay_rate
        )
        
        # Print layer-wise learning rates
        print("\nLayer-wise Learning Rates:")
        for i, group in enumerate(param_groups):
            layer_id = group.get('layer_id', -1)
            lr = group['lr']
            wd = group['weight_decay']
            num_params = len(group['params'])
            print(f"  Group {i} (Layer {layer_id}): LR={lr:.6f}, WD={wd:.4f}, Params={num_params}")
    else:
        # Standard parameter groups without layer decay
        param_groups = [
            {'params': [p for p in model.parameters() if p.requires_grad and not any(nd in n for nd in ['bias', 'norm', 'gamma'] for n, np in model.named_parameters() if np is p)],
             'weight_decay': config.training.optimizer.weight_decay},
            {'params': [p for p in model.parameters() if p.requires_grad and any(nd in n for nd in ['bias', 'norm', 'gamma'] for n, np in model.named_parameters() if np is p)],
             'weight_decay': 0.0}
        ]
    
    # Create optimizer
    if config.training.optimizer.type == 'AdamW':
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=config.training.optimizer.lr,
            betas=config.training.optimizer.betas,
            eps=config.training.optimizer.eps
        )
    else:
        raise ValueError(f"Unknown optimizer type: {config.training.optimizer.type}")
    
    # Create scheduler
    if config.training.scheduler.type == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        
        base_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs,
            eta_min=config.training.scheduler.min_lr
        )
        
        if config.training.layer_lr_decay.enabled:
            scheduler = LayerWiseLRScheduler(
                optimizer,
                base_scheduler,
                layer_decay=config.training.layer_lr_decay.decay_rate
            )
        else:
            scheduler = base_scheduler
    else:
        scheduler = None
    
    return optimizer, scheduler


def log_learning_rates(optimizer: torch.optim.Optimizer, epoch: int):
    """Log current learning rates for all parameter groups."""
    lrs = []
    for i, group in enumerate(optimizer.param_groups):
        layer_id = group.get('layer_id', -1)
        lr = group['lr']
        lrs.append(f"L{layer_id}:{lr:.2e}")
    
    print(f"Epoch {epoch} LRs: " + " | ".join(lrs))


if __name__ == "__main__":
    # Test layer-wise decay
    from models.convnext_tiny_bird import ConvNeXtTinyBird
    
    model = ConvNeXtTinyBird()
    
    # Test parameter grouping
    param_groups = get_parameter_groups(
        model,
        base_lr=1e-3,
        weight_decay=0.05,
        layer_decay=0.7
    )
    
    print(f"Created {len(param_groups)} parameter groups")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(param_groups)
    
    # Test scheduler
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=100,
        eta_min=1e-6
    )
    
    scheduler = LayerWiseLRScheduler(optimizer, base_scheduler, layer_decay=0.7)
    
    # Test stepping
    for epoch in range(5):
        scheduler.step()
        log_learning_rates(optimizer, epoch)
