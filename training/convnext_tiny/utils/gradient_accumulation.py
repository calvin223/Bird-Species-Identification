"""
Gradient accumulation utilities for effective larger batch sizes.
Critical for training with limited GPU memory.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


class GradientAccumulator:
    """Handles gradient accumulation for larger effective batch sizes."""
    
    def __init__(self, 
                 model: nn.Module,
                 accumulation_steps: int = 4,
                 max_grad_norm: float = 1.0):
        """
        Args:
            model: The model to accumulate gradients for
            accumulation_steps: Number of steps to accumulate
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.model = model
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        
    def should_step(self) -> bool:
        """Check if optimizer should step."""
        return (self.step_count + 1) % self.accumulation_steps == 0
    
    def increment(self):
        """Increment step counter."""
        self.step_count += 1
    
    def reset(self):
        """Reset step counter."""
        self.step_count = 0
    
    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss for gradient accumulation."""
        return loss / self.accumulation_steps
    
    def clip_gradients(self):
        """Clip gradients to prevent exploding gradients."""
        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
    
    def get_effective_batch_size(self, actual_batch_size: int) -> int:
        """Calculate effective batch size."""
        return actual_batch_size * self.accumulation_steps


class MixedPrecisionGradientAccumulator(GradientAccumulator):
    """Gradient accumulation with mixed precision support."""
    
    def __init__(self,
                 model: nn.Module,
                 accumulation_steps: int = 4,
                 max_grad_norm: float = 1.0,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None):
        super().__init__(model, accumulation_steps, max_grad_norm)
        self.scaler = scaler or torch.cuda.amp.GradScaler()
        
    def backward(self, loss: torch.Tensor):
        """Backward pass with mixed precision."""
        scaled_loss = self.scale_loss(loss)
        self.scaler.scale(scaled_loss).backward()
    
    def step(self, optimizer: torch.optim.Optimizer) -> bool:
        """Optimizer step with gradient clipping and scaling."""
        if self.should_step():
            # Unscale gradients
            self.scaler.unscale_(optimizer)
            
            # Clip gradients
            self.clip_gradients()
            
            # Step optimizer
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            return True
        
        self.increment()
        return False


class DynamicAccumulator(MixedPrecisionGradientAccumulator):
    """Dynamic gradient accumulation that adjusts based on memory usage."""
    
    def __init__(self,
                 model: nn.Module,
                 initial_accumulation_steps: int = 4,
                 min_accumulation_steps: int = 1,
                 max_accumulation_steps: int = 16,
                 max_grad_norm: float = 1.0,
                 scaler: Optional[torch.cuda.amp.GradScaler] = None,
                 memory_threshold_gb: float = 3.5):
        super().__init__(model, initial_accumulation_steps, max_grad_norm, scaler)
        self.min_accumulation_steps = min_accumulation_steps
        self.max_accumulation_steps = max_accumulation_steps
        self.memory_threshold_gb = memory_threshold_gb
        self.initial_accumulation_steps = initial_accumulation_steps
        
    def adjust_accumulation_steps(self, current_memory_gb: float):
        """Adjust accumulation steps based on memory usage."""
        if current_memory_gb > self.memory_threshold_gb:
            # Increase accumulation steps to reduce memory
            new_steps = min(
                self.accumulation_steps * 2,
                self.max_accumulation_steps
            )
            if new_steps != self.accumulation_steps:
                print(f"Increasing accumulation steps: {self.accumulation_steps} -> {new_steps}")
                self.accumulation_steps = new_steps
        elif current_memory_gb < self.memory_threshold_gb * 0.7:
            # Decrease accumulation steps if memory allows
            new_steps = max(
                self.accumulation_steps // 2,
                self.min_accumulation_steps
            )
            if new_steps != self.accumulation_steps:
                print(f"Decreasing accumulation steps: {self.accumulation_steps} -> {new_steps}")
                self.accumulation_steps = new_steps
    
    def reset_to_initial(self):
        """Reset to initial accumulation steps."""
        self.accumulation_steps = self.initial_accumulation_steps
        self.reset()


def create_gradient_accumulator(model: nn.Module, 
                              config: Dict[str, Any],
                              scaler: Optional[torch.cuda.amp.GradScaler] = None) -> GradientAccumulator:
    """Create gradient accumulator from config."""
    
    if config.get('memory', {}).get('auto_reduce_batch', False):
        # Use dynamic accumulator
        return DynamicAccumulator(
            model=model,
            initial_accumulation_steps=config['training']['accumulation_steps'],
            min_accumulation_steps=1,
            max_accumulation_steps=16,
            max_grad_norm=1.0,
            scaler=scaler,
            memory_threshold_gb=config['memory']['warning_threshold_gb']
        )
    else:
        # Use standard accumulator with mixed precision
        return MixedPrecisionGradientAccumulator(
            model=model,
            accumulation_steps=config['training']['accumulation_steps'],
            max_grad_norm=1.0,
            scaler=scaler
        )


class AccumulationContext:
    """Context manager for gradient accumulation."""
    
    def __init__(self, accumulator: GradientAccumulator, optimizer: torch.optim.Optimizer):
        self.accumulator = accumulator
        self.optimizer = optimizer
        self.did_step = False
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            # Normal exit - check if we should step
            self.did_step = self.accumulator.step(self.optimizer)


def compute_gradient_stats(model: nn.Module) -> Dict[str, float]:
    """Compute gradient statistics for monitoring."""
    total_norm = 0.0
    param_count = 0
    min_grad = float('inf')
    max_grad = float('-inf')
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            param_count += 1
            
            grad_abs = p.grad.data.abs()
            min_grad = min(min_grad, grad_abs.min().item())
            max_grad = max(max_grad, grad_abs.max().item())
    
    total_norm = total_norm ** 0.5
    
    return {
        'grad_norm': total_norm,
        'grad_min': min_grad if min_grad != float('inf') else 0.0,
        'grad_max': max_grad if max_grad != float('-inf') else 0.0,
        'param_count': param_count
    }


if __name__ == "__main__":
    # Test gradient accumulation
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = torch.cuda.amp.GradScaler()
    
    # Create accumulator
    accumulator = MixedPrecisionGradientAccumulator(
        model=model,
        accumulation_steps=4,
        scaler=scaler
    )
    
    # Simulate training loop
    for i in range(8):
        # Forward pass
        x = torch.randn(2, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward with accumulation
        accumulator.backward(loss)
        
        # Step optimizer if needed
        did_step = accumulator.step(optimizer)
        
        print(f"Step {i}: did_step={did_step}, accumulation_count={accumulator.step_count}")
