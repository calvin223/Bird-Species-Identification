"""
Main training script for ConvNeXt-Tiny on bird call classification.
Optimized for GTX 1650 (4GB VRAM) with aggressive memory management.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.cuda.amp as amp
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
import time
import argparse
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import Dict, Tuple, Optional
import json
from omegaconf import OmegaConf

# Add project root to path
project_root = Path(__file__).parent.absolute()
sys.path.append(str(project_root))

# Import custom modules
from models.convnext_tiny_bird import create_model, count_parameters
from utils.memory_efficient_loader import create_data_loaders
from utils.augmentations_convnext import create_augmentation_pipeline
from utils.gradient_accumulation import create_gradient_accumulator
from utils.memory_monitor import create_memory_monitor, MemoryProfiler, log_memory_usage
from utils.layer_lr_decay import create_optimizer_with_layer_decay, log_learning_rates

warnings.filterwarnings("ignore", category=UserWarning)


class ConvNeXtTrainer:
    """Trainer for ConvNeXt-Tiny with memory optimization."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create directories
        os.makedirs(config.training.checkpoint.save_dir, exist_ok=True)
        os.makedirs(config.logging.log_dir, exist_ok=True)
        
        # Initialize components
        self._setup_model()
        self._setup_data()
        self._setup_training()
        self._setup_monitoring()
        
        # Training state
        self.start_epoch = 0
        self.best_accuracy = 0.0
        self.best_epoch = 0
        
    def _setup_model(self):
        """Initialize model."""
        print("Setting up ConvNeXt-Tiny model...")
        self.model = create_model(self.config)
        self.model = self.model.to(self.device)
        
        # Model info
        param_count = count_parameters(self.model)
        print(f"Model parameters: {param_count:,}")
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        print(f"Model size: {param_size:.2f} MB")
        
    def _setup_data(self):
        """Initialize data loaders."""
        print("Setting up data loaders...")
        
        # Create augmentation pipeline
        self.train_transform = create_augmentation_pipeline(self.config)
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader, self.train_sampler = create_data_loaders(
            self.config,
            transform_train=self.train_transform,
            transform_val=None  # No augmentation for validation
        )
        
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Test batches: {len(self.test_loader)}")
        
    def _setup_training(self):
        """Initialize training components."""
        print("Setting up training components...")
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=self.config.training.loss.label_smoothing
        )
        
        # Optimizer with layer-wise LR decay
        self.optimizer, self.scheduler = create_optimizer_with_layer_decay(
            self.model, 
            self.config
        )
        
        # Mixed precision
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.config.training.mixed_precision.enabled)
        
        # Gradient accumulation
        self.accumulator = create_gradient_accumulator(
            self.model,
            self.config,
            self.scaler
        )
        
        # Class weights for imbalanced data
        if self.config.training.loss.use_class_weights:
            self._compute_class_weights()
        
    def _setup_monitoring(self):
        """Initialize monitoring tools."""
        print("Setting up monitoring...")
        
        # Memory monitor
        self.memory_monitor = create_memory_monitor(self.config)
        self.memory_profiler = MemoryProfiler(enabled=True)
        
        # TensorBoard
        if self.config.logging.tensorboard:
            self.writer = SummaryWriter(
                os.path.join(self.config.logging.log_dir, 
                           f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            )
        else:
            self.writer = None
            
        # Metrics tracking
        self.metrics_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': [],
            'memory_usage': []
        }
        
    def _compute_class_weights(self):
        """Compute class weights for imbalanced dataset."""
        # Count samples per class from metadata
        samples_per_class = {
            0: 2468, 1: 3325, 2: 3087, 3: 3954, 4: 3879,
            5: 4160, 6: 4116, 7: 3931, 8: 4020, 9: 3711,
            10: 3707, 11: 3845, 12: 3911, 13: 3602, 14: 4141,
            15: 3401, 16: 3788, 17: 4101, 18: 4160, 19: 4169,
            20: 3264, 21: 4118, 22: 2729, 23: 2938, 24: 4147,
            25: 3514, 26: 3628, 27: 3142, 28: 4165, 29: 3912
        }
        
        total_samples = sum(samples_per_class.values())
        num_classes = len(samples_per_class)
        
        # Compute weights
        weights = []
        for i in range(num_classes):
            weight = total_samples / (num_classes * samples_per_class[i])
            weights.append(weight)
        
        self.class_weights = torch.tensor(weights, device=self.device, dtype=torch.float32)
        
        # Update criterion
        self.criterion = nn.CrossEntropyLoss(
            weight=self.class_weights,
            label_smoothing=self.config.training.loss.label_smoothing
        )
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        self.train_transform.enable_mix_augmentations()
        
        # Set epoch for sampler
        if hasattr(self.train_sampler, 'set_epoch'):
            self.train_sampler.set_epoch(epoch)
        
        # Metrics
        running_loss = 0.0
        correct = 0
        total = 0
        batch_times = []
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}', 
                   total=len(self.train_loader), leave=False)
        
        for batch_idx, (data, target) in enumerate(pbar):
            batch_start = time.time()
            
            # Move to device
            data = data.to(self.device, non_blocking=True)
            
            # Handle MixUp/CutMix targets
            if isinstance(target, tuple):
                target_a, target_b, lam = target
                target_a = target_a.to(self.device, non_blocking=True)
                target_b = target_b.to(self.device, non_blocking=True)
                mixed_target = True
            else:
                target = target.to(self.device, non_blocking=True)
                mixed_target = False
            
            # Memory check
            if batch_idx % self.config.memory.monitor_interval == 0:
                status = self.memory_monitor.check_memory()
                if status == 'critical':
                    self.memory_monitor.optimize_memory()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config.training.mixed_precision.enabled):
                with self.memory_profiler.profile('forward'):
                    output = self.model(data)
                
                # Compute loss
                if mixed_target:
                    loss = lam * self.criterion(output, target_a) + \
                           (1 - lam) * self.criterion(output, target_b)
                else:
                    loss = self.criterion(output, target)
            
            # Backward pass with gradient accumulation
            self.accumulator.backward(loss)
            
            # Update weights if needed
            if self.accumulator.step(self.optimizer):
                # Clear cache periodically
                if batch_idx % self.config.memory.clear_cache_interval == 0:
                    self.memory_monitor.clear_cache()
            
            # Update metrics
            running_loss += loss.item()
            
            if not mixed_target:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            
            if total > 0:
                acc = 100. * correct / total
            else:
                acc = 0.0
            
            pbar.set_postfix({
                'loss': f'{running_loss/(batch_idx+1):.4f}',
                'acc': f'{acc:.2f}%',
                'mem': f'{self.memory_monitor.get_gpu_memory_stats()["allocated_gb"]:.2f}GB'
            })
            
            # Log to TensorBoard
            if self.writer and batch_idx % self.config.logging.log_interval == 0:
                global_step = epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('train/batch_loss', loss.item(), global_step)
                self.writer.add_scalar('train/batch_time', batch_time, global_step)
        
        # Epoch statistics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total if total > 0 else 0.0
        avg_batch_time = np.mean(batch_times)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'avg_batch_time': avg_batch_time
        }
    
    def validate(self, loader, desc: str = 'Validation') -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        self.train_transform.disable_mix_augmentations()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class statistics
        class_correct = [0] * self.config.dataset.num_classes
        class_total = [0] * self.config.dataset.num_classes
        
        # Confusion matrix
        confusion_matrix = torch.zeros(
            self.config.dataset.num_classes, 
            self.config.dataset.num_classes,
            dtype=torch.long
        )
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc, leave=False)
            
            for data, target in pbar:
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                
                # Forward pass
                with torch.amp.autocast('cuda', enabled=self.config.training.mixed_precision.enabled):
                    output = self.model(data)
                    loss = self.criterion(output, target)
                
                # Metrics
                running_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Per-class accuracy
                for t, p in zip(target.view(-1), predicted.view(-1)):
                    class_total[t.item()] += 1
                    if t == p:
                        class_correct[t.item()] += 1
                    confusion_matrix[t.item(), p.item()] += 1
                
                # Update progress
                acc = 100. * correct / total
                pbar.set_postfix({'loss': f'{running_loss/len(pbar):.4f}', 'acc': f'{acc:.2f}%'})
        
        # Calculate metrics
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        # Per-class accuracy
        per_class_acc = {}
        for i in range(self.config.dataset.num_classes):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                per_class_acc[self.config.dataset.class_names[i]] = acc
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'per_class_accuracy': per_class_acc,
            'confusion_matrix': confusion_matrix
        }
    
    def train(self):
        """Main training loop."""
        print(f"\nStarting training for {self.config.training.num_epochs} epochs...")
        print(f"Effective batch size: {self.accumulator.get_effective_batch_size(self.config.training.batch_size)}")
        
        # Log initial memory state
        self.memory_monitor.print_memory_summary()
        
        # Training loop
        for epoch in range(self.start_epoch, self.config.training.num_epochs):
            epoch_start = time.time()
            
            # Learning rate warmup
            if epoch < self.config.training.warmup_epochs:
                warmup_factor = (epoch + 1) / self.config.training.warmup_epochs
                warmup_lr = self.config.training.scheduler.warmup_lr + \
                           (self.config.training.optimizer.lr - self.config.training.scheduler.warmup_lr) * warmup_factor
                for group in self.optimizer.param_groups:
                    group['lr'] = warmup_lr
            
            # Log learning rates
            if epoch % 10 == 0:
                log_learning_rates(self.optimizer, epoch)
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(self.val_loader, 'Validation')
            
            # Step scheduler
            if self.scheduler and epoch >= self.config.training.warmup_epochs:
                self.scheduler.step()
            
            # Log metrics
            self.metrics_history['train_loss'].append(train_metrics['loss'])
            self.metrics_history['train_acc'].append(train_metrics['accuracy'])
            self.metrics_history['val_loss'].append(val_metrics['loss'])
            self.metrics_history['val_acc'].append(val_metrics['accuracy'])
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{self.config.training.num_epochs-1}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            print(f"  Time: {epoch_time:.1f}s ({train_metrics['avg_batch_time']:.3f}s/batch)")
            print(f"  Memory: {self.memory_monitor.get_gpu_memory_stats()['allocated_gb']:.2f}GB")
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalars('loss', {
                    'train': train_metrics['loss'],
                    'val': val_metrics['loss']
                }, epoch)
                self.writer.add_scalars('accuracy', {
                    'train': train_metrics['accuracy'],
                    'val': val_metrics['accuracy']
                }, epoch)
                
                # Log per-class accuracy
                for class_name, acc in val_metrics['per_class_accuracy'].items():
                    self.writer.add_scalar(f'val_acc_class/{class_name}', acc, epoch)
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_accuracy
            if is_best:
                self.best_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch
                
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Early stopping check
            if epoch - self.best_epoch > self.config.training.early_stopping.patience:
                print(f"\nEarly stopping triggered! Best accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
                break
        
        # Final summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        print(f"Best validation accuracy: {self.best_accuracy:.2f}% at epoch {self.best_epoch}")
        
        # Memory profiling summary
        self.memory_profiler.print_summary()
        self.memory_monitor.print_memory_summary()
        
        # Test on best model
        self.load_checkpoint('best')
        test_metrics = self.validate(self.test_loader, 'Test')
        print(f"\nTest accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Save final results
        self.save_results(test_metrics)
        
        if self.writer:
            self.writer.close()
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'best_accuracy': self.best_accuracy,
            'metrics': metrics,
            'config': OmegaConf.to_container(self.config)
        }
        
        # Save latest
        path = os.path.join(self.config.training.checkpoint.save_dir, 'latest.pth')
        torch.save(checkpoint, path, _use_new_zipfile_serialization=True)
        
        # Save best
        if is_best:
            path = os.path.join(self.config.training.checkpoint.save_dir, 'best.pth')
            torch.save(checkpoint, path)
            
        # Save periodic checkpoints
        if epoch % 50 == 0:
            path = os.path.join(self.config.training.checkpoint.save_dir, f'epoch_{epoch}.pth')
            torch.save(checkpoint, path)
    
    def load_checkpoint(self, checkpoint_name: str = 'latest'):
        """Load model checkpoint."""
        path = os.path.join(self.config.training.checkpoint.save_dir, f'{checkpoint_name}.pth')
        if os.path.exists(path):
            print(f"Loading checkpoint from {path}")
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.scheduler and checkpoint['scheduler_state_dict']:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_accuracy = checkpoint['best_accuracy']
            
            print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        else:
            print(f"No checkpoint found at {path}")
    
    def save_results(self, test_metrics: Dict):
        """Save final results."""
        results = {
            'model': 'ConvNeXt-Tiny',
            'best_val_accuracy': self.best_accuracy,
            'best_epoch': self.best_epoch,
            'test_accuracy': test_metrics['accuracy'],
            'test_loss': test_metrics['loss'],
            'per_class_accuracy': test_metrics['per_class_accuracy'],
            'config': OmegaConf.to_container(self.config),
            'metrics_history': self.metrics_history,
            'memory_peak_gb': self.memory_monitor.peak_memory_gb
        }
        
        # Save JSON
        with open(os.path.join(self.config.training.checkpoint.save_dir, 'results.json'), 'w') as f:
            json.dump(results, f, indent=4)
        
        # Save confusion matrix
        np.save(
            os.path.join(self.config.training.checkpoint.save_dir, 'confusion_matrix.npy'),
            test_metrics['confusion_matrix'].cpu().numpy()
        )
        
        # Print challenging species performance
        print("\nChallenging species performance:")
        for species in self.config.dataset.challenging_species:
            if species in test_metrics['per_class_accuracy']:
                acc = test_metrics['per_class_accuracy'][species]
                print(f"  {species}: {acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Train ConvNeXt-Tiny on bird calls')
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Resume from checkpoint')
    parser.add_argument('--test-only', action='store_true',
                      help='Only run test evaluation')
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create trainer
    trainer = ConvNeXtTrainer(config)
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train or test
    if args.test_only:
        trainer.load_checkpoint('best')
        test_metrics = trainer.validate(trainer.test_loader, 'Test')
        print(f"Test accuracy: {test_metrics['accuracy']:.2f}%")
        trainer.save_results(test_metrics)
    else:
        trainer.train()


if __name__ == "__main__":
    main()
