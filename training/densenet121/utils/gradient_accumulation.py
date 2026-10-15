"""
Gradient accumulation utilities for effective larger batch sizes on limited VRAM.
"""

import torch
from typing import Optional, Callable, Tuple
import logging
import random


class GradientAccumulator:
    """Handles gradient accumulation for training with limited memory."""
    
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4,
                 max_grad_norm: Optional[float] = 1.0,
                 mixed_precision: bool = True):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
        # Tracking
        self.step_count = 0
        self.accumulated_loss = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    def accumulate_step(self, 
                       loss: torch.Tensor,
                       retain_graph: bool = False) -> bool:
        """
        Accumulate gradients from a single step.
        
        Args:
            loss: Loss tensor from forward pass
            retain_graph: Whether to retain computation graph
            
        Returns:
            True if optimizer step was performed
        """
        # Scale loss by accumulation steps
        scaled_loss = loss / self.accumulation_steps
        
        # Backward pass
        if self.mixed_precision and self.scaler is not None:
            self.scaler.scale(scaled_loss).backward(retain_graph=retain_graph)
        else:
            scaled_loss.backward(retain_graph=retain_graph)
        
        self.accumulated_loss += loss.item()
        self.step_count += 1
        
        # Check if we should update weights
        if self.step_count % self.accumulation_steps == 0:
            self._optimizer_step()
            return True
        
        return False
    
    def _optimizer_step(self):
        """Perform optimizer step with gradient clipping."""
        if self.mixed_precision and self.scaler is not None:
            # Unscale gradients
            self.scaler.unscale_(self.optimizer)
            
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Gradient clipping
            if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.max_grad_norm
                )
            
            # Optimizer step
            self.optimizer.step()
        
        # Clear gradients
        self.optimizer.zero_grad(set_to_none=True)
        
        # Log average loss
        avg_loss = self.accumulated_loss / self.accumulation_steps
        self.logger.debug(f"Optimizer step - Average loss: {avg_loss:.4f}")
        
        # Reset tracking
        self.accumulated_loss = 0.0
    
    def finish_epoch(self):
        """Handle any remaining gradients at epoch end."""
        if self.step_count % self.accumulation_steps != 0:
            self.logger.warning(
                f"Epoch ended with {self.step_count % self.accumulation_steps} "
                f"unprocessed gradient accumulation steps"
            )
            # Perform final optimizer step
            self._optimizer_step()
        
        self.step_count = 0
        self.accumulated_loss = 0.0
    
    def get_effective_batch_size(self, actual_batch_size: int) -> int:
        """Calculate effective batch size with accumulation."""
        return actual_batch_size * self.accumulation_steps


class MemoryEfficientTrainingLoop:
    """Training loop with gradient accumulation and memory management."""
    
    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 accumulation_steps: int = 4,
                 mixed_precision: bool = True,
                 gradient_clipping: float = 1.0):
        
        self.model = model
        self.criterion = criterion
        self.accumulator = GradientAccumulator(
            model=model,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
            max_grad_norm=gradient_clipping,
            mixed_precision=mixed_precision
        )
        
        self.mixed_precision = mixed_precision
        self.device = next(model.parameters()).device
    
    def train_step(self,
                   inputs: torch.Tensor,
                   targets: torch.Tensor,
                   mixup_fn: Optional[Callable] = None) -> float:
        """
        Single training step with gradient accumulation.
        
        Returns:
            Loss value for this step
        """
        self.model.train()
        
        # Apply mixup if provided
        if mixup_fn is not None:
            inputs, targets_a, targets_b, lam = mixup_fn(inputs, targets)
        
        # Forward pass with mixed precision
        if self.mixed_precision:
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                
                if mixup_fn is not None:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
        else:
            outputs = self.model(inputs)
            
            if mixup_fn is not None:
                loss = lam * self.criterion(outputs, targets_a) + \
                       (1 - lam) * self.criterion(outputs, targets_b)
            else:
                loss = self.criterion(outputs, targets)
        
        # Accumulate gradients
        optimizer_stepped = self.accumulator.accumulate_step(loss)
        
        # Clear intermediate tensors to free memory
        del outputs
        if mixup_fn is not None:
            del targets_a, targets_b
        
        return loss.item()
    
    def validation_step(self,
                       inputs: torch.Tensor,
                       targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Single validation step without gradient computation.
        
        Returns:
            Loss value and predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
        
        return loss.item(), outputs


def create_balanced_sampler(dataset_indices: dict, 
                           batch_size: int,
                           accumulation_steps: int = 4):
    """
    Create a balanced batch sampler for gradient accumulation.
    
    Args:
        dataset_indices: Dict mapping class indices to sample indices
        batch_size: Actual batch size (before accumulation)
        accumulation_steps: Number of accumulation steps
    
    Returns:
        Batch sampler that ensures balanced classes
    """
    # Calculate samples per class in effective batch
    effective_batch_size = batch_size * accumulation_steps
    num_classes = len(dataset_indices)
    samples_per_class = max(1, effective_batch_size // num_classes)
    
    class BalancedBatchSampler:
        def __init__(self, dataset_indices, batch_size, samples_per_class):
            self.dataset_indices = dataset_indices
            self.batch_size = batch_size
            self.samples_per_class = samples_per_class
            self.num_classes = len(dataset_indices)
            
            # Calculate total batches
            min_class_size = min(len(indices) for indices in dataset_indices.values())
            self.num_batches = min_class_size // samples_per_class
        
        def __iter__(self):
            # Shuffle indices for each class
            shuffled_indices = {}
            for class_idx, indices in self.dataset_indices.items():
                shuffled = torch.randperm(len(indices)).tolist()
                shuffled_indices[class_idx] = [indices[i] for i in shuffled]
            
            # Generate balanced batches
            for batch_idx in range(self.num_batches):
                batch = []
                for class_idx in range(self.num_classes):
                    start_idx = batch_idx * self.samples_per_class
                    end_idx = start_idx + self.samples_per_class
                    batch.extend(shuffled_indices[class_idx][start_idx:end_idx])
                
                # Shuffle within batch
                random.shuffle(batch)
                
                # Yield in chunks of actual batch size
                for i in range(0, len(batch), self.batch_size):
                    yield batch[i:i + self.batch_size]
        
        def __len__(self):
            return self.num_batches * (self.num_classes * self.samples_per_class // self.batch_size)
    
    return BalancedBatchSampler(dataset_indices, batch_size, samples_per_class)
