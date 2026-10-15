"""
Memory monitoring utilities for GTX 1650 (4GB VRAM).
Tracks GPU memory usage and provides alerts/adjustments.
"""

import torch
import psutil
import gc
import warnings

# Try to import GPUtil, but make it optional
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False
    print("Warning: GPUtil not installed. GPU memory stats will be limited.")
from typing import Dict, Optional, Callable
import time

warnings.filterwarnings("ignore", category=UserWarning)


class MemoryMonitor:
    """Monitor GPU and system memory usage."""
    
    def __init__(self, 
                 max_gpu_memory_gb: float = 3.8,
                 warning_threshold_gb: float = 3.5,
                 critical_threshold_gb: float = 3.7,
                 device: str = 'cuda'):
        """
        Args:
            max_gpu_memory_gb: Maximum GPU memory to use
            warning_threshold_gb: Threshold for warnings
            critical_threshold_gb: Threshold for critical actions
            device: Device to monitor
        """
        self.max_gpu_memory_gb = max_gpu_memory_gb
        self.warning_threshold_gb = warning_threshold_gb
        self.critical_threshold_gb = critical_threshold_gb
        self.device = device
        
        # History tracking
        self.memory_history = []
        self.peak_memory_gb = 0.0
        
        # Callbacks
        self.warning_callback = None
        self.critical_callback = None
        
    def get_gpu_memory_stats(self) -> Dict[str, float]:
        """Get current GPU memory statistics."""
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'free_gb': 0.0,
                'total_gb': 0.0
            }
        
        # Get memory stats in GB
        allocated = torch.cuda.memory_allocated(self.device) / 1024**3
        reserved = torch.cuda.memory_reserved(self.device) / 1024**3
        
        # Get total GPU memory
        if GPUTIL_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                total = gpus[0].memoryTotal / 1024  # Convert MB to GB
                free = gpus[0].memoryFree / 1024
            else:
                total = 4.0  # Default for GTX 1650
                free = total - reserved
        else:
            # Fallback when GPUtil is not available
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            free = (total * 1024**3 - torch.cuda.memory_reserved(self.device)) / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'total_gb': total,
            'usage_percent': (allocated / total) * 100
        }
    
    def get_system_memory_stats(self) -> Dict[str, float]:
        """Get system RAM statistics."""
        vm = psutil.virtual_memory()
        return {
            'total_gb': vm.total / 1024**3,
            'available_gb': vm.available / 1024**3,
            'used_gb': vm.used / 1024**3,
            'percent': vm.percent
        }
    
    def check_memory(self) -> str:
        """Check memory and return status."""
        gpu_stats = self.get_gpu_memory_stats()
        allocated = gpu_stats['allocated_gb']
        
        # Update peak memory
        self.peak_memory_gb = max(self.peak_memory_gb, allocated)
        
        # Add to history
        self.memory_history.append({
            'timestamp': time.time(),
            'allocated_gb': allocated,
            'reserved_gb': gpu_stats['reserved_gb']
        })
        
        # Check thresholds
        if allocated > self.critical_threshold_gb:
            if self.critical_callback:
                self.critical_callback(gpu_stats)
            return 'critical'
        elif allocated > self.warning_threshold_gb:
            if self.warning_callback:
                self.warning_callback(gpu_stats)
            return 'warning'
        
        return 'ok'
    
    def clear_cache(self):
        """Clear GPU cache and run garbage collection."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def optimize_memory(self):
        """Aggressive memory optimization."""
        # Clear cache
        self.clear_cache()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
        
        # Clear Python's internal caches
        if hasattr(gc, 'freeze'):
            gc.freeze()
            gc.collect()
            gc.unfreeze()
    
    def print_memory_summary(self):
        """Print memory usage summary."""
        gpu_stats = self.get_gpu_memory_stats()
        sys_stats = self.get_system_memory_stats()
        
        print("\n" + "="*50)
        print("MEMORY USAGE SUMMARY")
        print("="*50)
        print(f"GPU Memory:")
        print(f"  Allocated: {gpu_stats['allocated_gb']:.2f} GB")
        print(f"  Reserved:  {gpu_stats['reserved_gb']:.2f} GB")
        print(f"  Free:      {gpu_stats['free_gb']:.2f} GB")
        print(f"  Total:     {gpu_stats['total_gb']:.2f} GB")
        print(f"  Usage:     {gpu_stats['usage_percent']:.1f}%")
        print(f"  Peak:      {self.peak_memory_gb:.2f} GB")
        print(f"\nSystem RAM:")
        print(f"  Used:      {sys_stats['used_gb']:.2f} GB")
        print(f"  Available: {sys_stats['available_gb']:.2f} GB")
        print(f"  Total:     {sys_stats['total_gb']:.2f} GB")
        print(f"  Usage:     {sys_stats['percent']:.1f}%")
        print("=" * 50 + "\n")
    
    def set_warning_callback(self, callback: Callable):
        """Set callback for memory warnings."""
        self.warning_callback = callback
    
    def set_critical_callback(self, callback: Callable):
        """Set callback for critical memory situations."""
        self.critical_callback = callback
    
    def get_memory_history(self, last_n: Optional[int] = None) -> list:
        """Get memory usage history."""
        if last_n:
            return self.memory_history[-last_n:]
        return self.memory_history


class AutoMemoryManager:
    """Automatically manage memory based on usage."""
    
    def __init__(self,
                 monitor: MemoryMonitor,
                 initial_batch_size: int = 8,
                 min_batch_size: int = 4):
        self.monitor = monitor
        self.initial_batch_size = initial_batch_size
        self.current_batch_size = initial_batch_size
        self.min_batch_size = min_batch_size
        
        # Track reductions
        self.reduction_count = 0
        self.last_reduction_time = 0
        
    def should_reduce_batch_size(self) -> bool:
        """Check if batch size should be reduced."""
        status = self.monitor.check_memory()
        
        if status == 'critical':
            # Don't reduce too frequently
            current_time = time.time()
            if current_time - self.last_reduction_time > 60:  # 1 minute cooldown
                self.last_reduction_time = current_time
                return True
        
        return False
    
    def reduce_batch_size(self) -> int:
        """Reduce batch size and return new size."""
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(
                self.min_batch_size,
                self.current_batch_size // 2
            )
            self.reduction_count += 1
            print(f"\nReducing batch size to {self.current_batch_size} due to memory pressure")
            
            # Clear memory after reduction
            self.monitor.optimize_memory()
            
        return self.current_batch_size
    
    def reset(self):
        """Reset to initial batch size."""
        self.current_batch_size = self.initial_batch_size
        self.reduction_count = 0


class MemoryProfiler:
    """Profile memory usage during training."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.profiles = {}
        
    def profile(self, name: str):
        """Context manager for profiling memory usage."""
        return MemoryProfileContext(name, self)
    
    def add_profile(self, name: str, start_mem: float, end_mem: float, duration: float):
        """Add profiling results."""
        if name not in self.profiles:
            self.profiles[name] = []
        
        self.profiles[name].append({
            'start_mem_gb': start_mem,
            'end_mem_gb': end_mem,
            'delta_gb': end_mem - start_mem,
            'duration_s': duration
        })
    
    def print_summary(self):
        """Print profiling summary."""
        if not self.profiles:
            return
        
        print("\n" + "="*60)
        print("MEMORY PROFILING SUMMARY")
        print("="*60)
        
        for name, records in self.profiles.items():
            if records:
                avg_delta = sum(r['delta_gb'] for r in records) / len(records)
                max_delta = max(r['delta_gb'] for r in records)
                avg_duration = sum(r['duration_s'] for r in records) / len(records)
                
                print(f"\n{name}:")
                print(f"  Average memory increase: {avg_delta:.3f} GB")
                print(f"  Maximum memory increase: {max_delta:.3f} GB")
                print(f"  Average duration: {avg_duration:.3f} s")
                print(f"  Number of calls: {len(records)}")


class MemoryProfileContext:
    """Context manager for memory profiling."""
    
    def __init__(self, name: str, profiler: MemoryProfiler):
        self.name = name
        self.profiler = profiler
        self.start_mem = 0.0
        self.start_time = 0.0
        
    def __enter__(self):
        if self.profiler.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_mem = torch.cuda.memory_allocated() / 1024**3
            self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profiler.enabled and torch.cuda.is_available():
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated() / 1024**3
            duration = time.time() - self.start_time
            
            self.profiler.add_profile(
                self.name,
                self.start_mem,
                end_mem,
                duration
            )


def create_memory_monitor(config) -> MemoryMonitor:
    """Create memory monitor from config."""
    return MemoryMonitor(
        max_gpu_memory_gb=config.memory.max_gpu_memory_gb,
        warning_threshold_gb=config.memory.warning_threshold_gb,
        critical_threshold_gb=config.memory.get('critical_threshold_gb', 
                                               config.memory.warning_threshold_gb + 0.2)
    )


def log_memory_usage(prefix: str = ""):
    """Quick function to log current memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix}GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")


if __name__ == "__main__":
    # Test memory monitor
    monitor = MemoryMonitor()
    
    # Check memory
    status = monitor.check_memory()
    print(f"Memory status: {status}")
    
    # Print summary
    monitor.print_memory_summary()
    
    # Test profiler
    profiler = MemoryProfiler()
    
    # Simulate some operations
    with profiler.profile("test_operation"):
        if torch.cuda.is_available():
            x = torch.randn(1000, 1000, device='cuda')
            time.sleep(0.1)
    
    profiler.print_summary()
