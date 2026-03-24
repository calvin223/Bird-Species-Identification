"""
Main training script for memory-efficient DenseNet-121 on bird call classification.
Optimized for NVIDIA GTX 1650 (4GB VRAM).
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from models.densenet_memory_efficient import create_densenet121, count_parameters
from utils.memory_utils import (
    MemoryMonitor, clear_memory, optimize_memory_allocation,
    MemoryEfficientCheckpoint, handle_oom_error
)
from utils.augmentations_lite import (
    get_training_augmentations, MixUpLite, normalize_batch
)
from utils.gradient_accumulation import (
    GradientAccumulator, create_balanced_sampler
)


class BirdCallDataset(Dataset):
    """Memory-mapped dataset for bird call spectrograms."""
    
    def __init__(self, 
                 data_path: str,
                 split: str = 'train',
                 transform=None,
                 config=None):
        self.data_path = Path(data_path)
        self.split = split
        self.transform = transform
        self.config = config
        
        # Load memory-mapped arrays
        # Check for different naming conventions
        data_file = self.data_path / f'{split}_data.dat'
        if not data_file.exists():
            data_file = self.data_path / f'{split}_features.dat'
        
        self.data = np.memmap(
            data_file,
            dtype='float32',
            mode='r',
            shape=self._get_data_shape()
        )
        
        # Check for labels file
        labels_file = self.data_path / f'{split}_labels.dat'
        if not labels_file.exists():
            # Try numpy format
            labels_file = self.data_path / f'{split}_labels.npy'
            if labels_file.exists():
                self.labels = np.load(labels_file)
            else:
                raise FileNotFoundError(f"Labels file not found for {split} split")
        else:
            self.labels = np.memmap(
                labels_file,
                dtype='int64',
                mode='r'
            )
        
        # Create class indices for balanced sampling
        self.class_indices = {}
        for idx in range(len(self.labels)):
            label = int(self.labels[idx])
            if label not in self.class_indices:
                self.class_indices[label] = []
            self.class_indices[label].append(idx)
        
        logging.info(f"Loaded {split} dataset: {len(self)} samples")
    
    def _get_data_shape(self):
        """Get shape of memory-mapped data."""
        # Read shape from metadata file
        metadata_path = self.data_path / f'{self.split}_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                # The shape is [num_samples, 1, 128, 259]
                # We need to return the full shape for memmap
                num_samples = metadata['num_samples']
                feature_shape = metadata['feature_shape']  # [1, 128, 259]
                return (num_samples, *feature_shape)
        
        # Check for shape file
        shape_file = self.data_path / f'{self.split}_features_shape.npy'
        if shape_file.exists():
            shape = np.load(shape_file)
            return tuple(shape)
        
        # Fallback: probe the file
        temp = np.memmap(
            self.data_path / f'{self.split}_features.dat',
            dtype='float32',
            mode='r'
        )
        # Assume shape based on known dimensions
        n_samples = len(temp) // (1 * 128 * 259)
        return (n_samples, 1, 128, 259)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load single spectrogram
        spectrogram = self.data[idx].copy()  # Copy to avoid memory map issues
        label = int(self.labels[idx])
        
        # Convert to tensor
        # spectrogram is already shape (1, 128, 259)
        spectrogram = torch.from_numpy(spectrogram)
        
        # Apply transforms
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))


class Trainer:
    """Main trainer class with memory management."""
    
    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Setup logging
        self._setup_logging()
        
        # Setup device and memory optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        optimize_memory_allocation()
        
        # Initialize memory monitor
        self.memory_monitor = MemoryMonitor(
            gpu_limit_gb=self.config['training']['memory_limit_gb']
        )
        
        # Create model
        self.model = create_densenet121(self.config).to(self.device)
        self.logger.info(f"Model parameters: {count_parameters(self.model)}")
        
        # Create datasets
        self._create_datasets()
        
        # Create optimizer and scheduler
        self._create_optimizer()
        
        # Create loss function
        self.criterion = LabelSmoothingCrossEntropy(
            smoothing=self.config['training']['label_smoothing']
        )
        
        # Initialize gradient accumulator
        self.accumulator = GradientAccumulator(
            model=self.model,
            optimizer=self.optimizer,
            accumulation_steps=self.config['training']['accumulation_steps'],
            max_grad_norm=self.config['training']['gradient_clipping'],
            mixed_precision=self.config['training']['mixed_precision']
        )
        
        # Initialize mixup
        self.mixup = MixUpLite(
            alpha=self.config['augmentation']['mixup']['alpha'],
            probability=self.config['augmentation']['mixup']['probability']
        ) if self.config['augmentation']['mixup']['enabled'] else None
        
        # Training state
        self.start_epoch = 0
        self.best_acc = 0.0
        self.early_stopping_counter = 0
        
        # Create checkpoint directory
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f'training_{datetime.now():%Y%m%d_%H%M%S}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_datasets(self):
        """Create train and validation datasets."""
        # Training augmentations
        train_transform = get_training_augmentations(self.config)
        
        # Create datasets
        self.train_dataset = BirdCallDataset(
            data_path=self.config['dataset']['data_path'],
            split='train',
            transform=train_transform,
            config=self.config
        )
        
        self.val_dataset = BirdCallDataset(
            data_path=self.config['dataset']['data_path'],
            split='val',
            transform=None,
            config=self.config
        )
        
        # Create data loaders with balanced sampling for training
        train_sampler = create_balanced_sampler(
            self.train_dataset.class_indices,
            self.config['training']['batch_size'],
            self.config['training']['accumulation_steps']
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=train_sampler,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'] * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=self.config['training']['num_workers'],
            pin_memory=self.config['training']['pin_memory']
        )
    
    def _create_optimizer(self):
        """Create optimizer and scheduler."""
        # SGD is more memory efficient than Adam
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            momentum=self.config['training']['momentum'],
            weight_decay=self.config['training']['weight_decay'],
            nesterov=self.config['training']['nesterov']
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode=self.config['scheduler']['mode'],
            patience=self.config['scheduler']['patience'],
            factor=self.config['scheduler']['factor'],
            min_lr=self.config['scheduler']['min_lr'],
            verbose=True
        )
    
    @handle_oom_error
    def train_epoch(self, epoch: int):
        """Train for one epoch with memory management."""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} Training')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Normalize batch
            inputs = normalize_batch(inputs)
            
            # Apply mixup
            if self.mixup is not None:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
            
            # Forward pass with mixed precision
            if self.config['training']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    if self.mixup is not None:
                        loss = lam * self.criterion(outputs, targets_a) + \
                               (1 - lam) * self.criterion(outputs, targets_b)
                    else:
                        loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                if self.mixup is not None:
                    loss = lam * self.criterion(outputs, targets_a) + \
                           (1 - lam) * self.criterion(outputs, targets_b)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Gradient accumulation
            self.accumulator.accumulate_step(loss)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if self.mixup is None:
                correct += predicted.eq(targets).sum().item()
            else:
                correct += (lam * predicted.eq(targets_a).float() + 
                           (1 - lam) * predicted.eq(targets_b).float()).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'gpu_mem': f"{self.memory_monitor.get_gpu_memory()['used']:.1f}GB"
            })
            
            # Memory monitoring
            if batch_idx % self.config['logging']['memory_log_interval'] == 0:
                if self.memory_monitor.check_memory_threshold():
                    self.logger.warning("GPU memory threshold exceeded!")
                    clear_memory()
            
            # Clear intermediate variables
            del outputs, loss
            if self.mixup is not None:
                del targets_a, targets_b
        
        # Finish any remaining gradients
        self.accumulator.finish_epoch()
        
        # Clear memory after epoch
        clear_memory()
        
        return total_loss / len(self.train_loader), 100. * correct / total
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model with memory efficiency."""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        class_correct = [0] * self.config['dataset']['num_classes']
        class_total = [0] * self.config['dataset']['num_classes']
        
        pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} Validation')
        
        for inputs, targets in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Normalize batch
            inputs = normalize_batch(inputs)
            
            # Forward pass
            if self.config['training']['mixed_precision']:
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Per-class accuracy
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_loss / (len(pbar) + 1),
                'acc': 100. * correct / total
            })
        
        # Calculate per-class accuracy
        class_accuracies = []
        for i in range(self.config['dataset']['num_classes']):
            if class_total[i] > 0:
                acc = 100. * class_correct[i] / class_total[i]
                class_accuracies.append(acc)
                self.logger.info(
                    f"Class {i} ({self.config['dataset']['species_names'][i]}): "
                    f"{acc:.2f}% ({class_correct[i]}/{class_total[i]})"
                )
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100. * correct / total
        
        # Clear memory
        clear_memory()
        
        return avg_loss, avg_acc, class_accuracies
    
    def train(self):
        """Main training loop."""
        self.logger.info("Starting training...")
        self.memory_monitor.log_memory_stats("Initial")
        
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc, class_accuracies = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Logging
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Save checkpoint
            is_best = val_acc > self.best_acc
            if is_best:
                self.best_acc = val_acc
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['training']['checkpoint_interval'] == 0:
                checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
                MemoryEfficientCheckpoint.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    best_acc=self.best_acc,
                    filepath=checkpoint_path
                )
            
            # Save best model
            if is_best:
                best_path = self.checkpoint_dir / 'best_model.pth'
                MemoryEfficientCheckpoint.save_checkpoint(
                    model=self.model,
                    optimizer=None,  # Don't save optimizer for best model
                    epoch=epoch + 1,
                    best_acc=self.best_acc,
                    filepath=best_path,
                    save_optimizer=False
                )
                self.logger.info(f"New best model saved with accuracy: {val_acc:.2f}%")
            
            # Early stopping
            if self.early_stopping_counter >= self.config['training']['early_stopping_patience']:
                self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Memory monitoring
            self.memory_monitor.log_memory_stats(f"Epoch {epoch+1}")
        
        self.logger.info(f"Training completed. Best accuracy: {self.best_acc:.2f}%")


def main():
    """Main entry point."""
    trainer = Trainer('config.json')
    trainer.train()


if __name__ == '__main__':
    main()
