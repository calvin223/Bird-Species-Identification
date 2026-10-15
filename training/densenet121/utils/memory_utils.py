"""
Memory management utilities for training on limited VRAM (4GB GTX 1650).
"""

import torch
import gc
import psutil
import GPUtil
import warnings
from typing import Dict, Optional, Tuple
import logging

class MemoryMonitor:
    """Monitor and manage GPU/CPU memory usage."""
    
    def __init__(self, gpu_limit_gb: float = 3.8):
        self.gpu_limit_gb = gpu_limit_gb
        self.gpu_limit_bytes = int(gpu_limit_gb * 1024 * 1024 * 1024)
        self.logger = logging.getLogger(__name__)
        
        # Check GPU availability
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu = GPUtil.getGPUs()[0]
            self.device_name = torch.cuda.get_device_name(0)
            self.logger.info(f"GPU detected: {self.device_name}")
        
    def get_gpu_memory(self) -> Dict[str, float]:
        """Get current GPU memory usage in GB."""
        if not self.gpu_available:
            return {"allocated": 0, "reserved": 0, "free": 0}
        
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        
        # Get total GPU memory
        gpu = GPUtil.getGPUs()[0]
        free = gpu.memoryFree / 1024  # Convert MB to GB
        total = gpu.memoryTotal / 1024
        used = gpu.memoryUsed / 1024
        
        return {
            "allocated": allocated,
            "reserved": reserved,
            "free": free,
            "used": used,
            "total": total,
            "percentage": (used / total) * 100
        }
    
    def get_cpu_memory(self) -> Dict[str, float]:
        """Get current CPU memory usage in GB."""
        memory = psutil.virtual_memory()
        return {
            "used": memory.used / 1024**3,
            "available": memory.available / 1024**3,
            "total": memory.total / 1024**3,
            "percentage": memory.percent
        }
    
    def check_memory_threshold(self) -> bool:
        """Check if GPU memory usage is approaching limit."""
        if not self.gpu_available:
            return False
        
        gpu_mem = self.get_gpu_memory()
        current_usage_gb = gpu_mem["used"]
        
        if current_usage_gb > self.gpu_limit_gb:
            self.logger.warning(
                f"GPU memory usage ({current_usage_gb:.2f}GB) exceeds limit ({self.gpu_limit_gb}GB)"
            )
            return True
        elif current_usage_gb > self.gpu_limit_gb * 0.9:
            self.logger.warning(
                f"GPU memory usage ({current_usage_gb:.2f}GB) approaching limit ({self.gpu_limit_gb}GB)"
            )
        
        return False
    
    def log_memory_stats(self, prefix: str = ""):
        """Log current memory statistics."""
        gpu_mem = self.get_gpu_memory()
        cpu_mem = self.get_cpu_memory()
        
        self.logger.info(
            f"{prefix} GPU Memory: {gpu_mem['used']:.2f}/{gpu_mem['total']:.2f}GB "
            f"({gpu_mem['percentage']:.1f}%), "
            f"CPU Memory: {cpu_mem['used']:.2f}/{cpu_mem['total']:.2f}GB "
            f"({cpu_mem['percentage']:.1f}%)"
        )


def clear_memory():
    """Aggressively clear GPU and CPU memory."""
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Clear Python objects
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                del obj
        except:
            pass
    
    gc.collect()


def optimize_memory_allocation():
    """Set PyTorch memory allocation settings for limited VRAM."""
    if torch.cuda.is_available():
        # Reduce memory fragmentation
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use up to 95% of VRAM
        
        # Enable cudnn benchmarking for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Set growth increment for memory allocator
        torch.cuda.empty_cache()


def estimate_batch_size(model: torch.nn.Module, 
                       input_shape: Tuple[int, ...],
                       max_memory_gb: float = 3.5) -> int:
    """Estimate maximum batch size that fits in memory."""
    if not torch.cuda.is_available():
        return 8
    
    device = torch.device("cuda")
    model = model.to(device)
    model.train()
    
    # Binary search for maximum batch size
    low, high = 1, 64
    working_batch_size = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Clear memory
            clear_memory()
            
            # Test forward and backward pass
            dummy_input = torch.randn(mid, *input_shape).to(device)
            dummy_target = torch.randint(0, 30, (mid,)).to(device)
            
            output = model(dummy_input)
            loss = torch.nn.functional.cross_entropy(output, dummy_target)
            loss.backward()
            
            # Check memory usage
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            
            if mem_gb < max_memory_gb * 0.8:  # 80% safety margin
                working_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
                
            # Clean up
            del dummy_input, dummy_target, output, loss
            clear_memory()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                high = mid - 1
                clear_memory()
            else:
                raise e
    
    model.zero_grad()
    clear_memory()
    
    return working_batch_size


class MemoryEfficientCheckpoint:
    """Memory-efficient model checkpointing."""
    
    @staticmethod
    def save_checkpoint(model: torch.nn.Module,
                       optimizer: torch.optim.Optimizer,
                       epoch: int,
                       best_acc: float,
                       filepath: str,
                       save_optimizer: bool = True):
        """Save checkpoint with minimal memory usage."""
        # Move model to CPU temporarily
        device = next(model.parameters()).device
        model.cpu()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_acc': best_acc,
        }
        
        if save_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        # Save with minimal memory
        torch.save(checkpoint, filepath, pickle_protocol=4)
        
        # Move model back to original device
        model.to(device)
        
        # Clear memory
        del checkpoint
        clear_memory()
    
    @staticmethod
    def load_checkpoint(filepath: str, 
                       model: torch.nn.Module,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       device: str = 'cuda'):
        """Load checkpoint with minimal memory usage."""
        # Load to CPU first
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Load optimizer if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Move optimizer state to correct device
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        
        epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0.0)
        
        # Clear memory
        del checkpoint
        clear_memory()
        
        return epoch, best_acc


def handle_oom_error(func):
    """Decorator to handle out-of-memory errors gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logging.error(f"OOM Error in {func.__name__}: {str(e)}")
                clear_memory()
                
                # Try with reduced batch size
                if 'batch_size' in kwargs:
                    new_batch_size = max(1, kwargs['batch_size'] // 2)
                    logging.info(f"Retrying with batch size: {new_batch_size}")
                    kwargs['batch_size'] = new_batch_size
                    return func(*args, **kwargs)
                else:
                    raise e
            else:
                raise e
    return wrapper
