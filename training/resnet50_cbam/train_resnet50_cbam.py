"""
ResNet-50 with CBAM Attention for Bird Call Classification
Advanced architecture with attention mechanisms for improved accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
from tqdm import tqdm
import wandb
from typing import Dict, Tuple, Optional
import json
from collections import defaultdict

from models.resnet50_cbam import create_model
from utils.data_loader import create_data_loaders, collate_fn_mixup
from utils.augmentations import MixUpCriterion, CutMix, CutMixCriterion
from utils.attention_viz import AttentionVisualizer


class Trainer:
    """Trainer class for ResNet-50 CBAM model"""
    
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.setup_logging()
        
        # Create model
        self.model = self.create_model()
        
        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = create_data_loaders(config)
        
        # Setup training components
        self.setup_training()
        
        # Initialize tracking
        self.best_val_accuracy = 0.0
        self.epoch = 0
        self.global_step = 0
        
        # Setup wandb if enabled
        if config['logging']['use_wandb']:
            self.setup_wandb()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path(self.config['logging']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'training_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Training started with config: {self.config}")
    
    def create_model(self) -> nn.Module:
        """Create and initialize model"""
        model = create_model(
            num_classes=self.config['model']['num_classes'],
            in_channels=self.config['model']['in_channels'],
            use_cbam=self.config['model']['use_cbam'],
            drop_path_rate=self.config['model']['drop_path_rate'],
            use_freq_attention=self.config['model']['use_freq_attention']
        )
        
        model = model.to(self.device)
        
        # Enable benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        self.logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def setup_training(self):
        """Setup training components"""
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['optimizer']['learning_rate'],
            weight_decay=self.config['optimizer']['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config['scheduler']['T_0'],
            T_mult=self.config['scheduler']['T_mult'],
            eta_min=float(self.config['scheduler']['eta_min'])
        )
        
        # Loss function
        if self.config['training']['use_label_smoothing']:
            self.criterion = nn.CrossEntropyLoss(
                label_smoothing=self.config['training']['label_smoothing_epsilon']
            )
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Move criterion to device
        self.criterion = self.criterion.to(self.device)
        
        # Mixed precision training
        self.scaler = GradScaler('cuda')
        
        # Augmentation modules
        if self.config['augmentation']['cutmix']['enabled']:
            self.cutmix = CutMix(
                alpha=self.config['augmentation']['cutmix']['alpha'],
                p=self.config['augmentation']['cutmix']['p']
            )
            self.cutmix_criterion = CutMixCriterion(self.criterion)
        
        if self.config['augmentation']['mixup']['enabled']:
            self.mixup_criterion = MixUpCriterion(self.criterion)
    
    def setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        wandb.init(
            project=self.config['logging']['wandb_project'],
            name=self.config['logging']['run_name'],
            config=self.config
        )
        wandb.watch(self.model, log_freq=100)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch + 1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            # Handle MixUp if using custom collate function
            if len(batch) == 4:  # MixUp batch
                inputs, targets, targets_b, lam = batch
                lam = lam.item()
                use_mixup = True
            else:
                inputs, targets = batch
                use_mixup = False
            
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Apply CutMix if enabled
            if hasattr(self, 'cutmix') and not use_mixup:
                inputs, targets_a, targets_b, lam = self.cutmix(inputs, targets)
                use_cutmix = True
            else:
                use_cutmix = False
            
            # Mixed precision forward pass
            with autocast('cuda'):
                outputs = self.model(inputs)
                
                # Calculate loss based on augmentation
                if use_mixup:
                    loss = self.mixup_criterion(outputs, targets, targets_b.to(self.device), lam)
                elif use_cutmix:
                    loss = self.cutmix_criterion(outputs, targets_a, targets_b, lam)
                else:
                    loss = self.criterion(outputs, targets)
            
            # Backward pass with gradient scaling
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          self.config['training']['gradient_clip_val'])
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            
            # For augmented samples, use weighted accuracy
            if use_mixup or use_cutmix:
                correct += (lam * predicted.eq(targets).sum().item() + 
                          (1 - lam) * predicted.eq(targets_b).sum().item())
            else:
                correct += predicted.eq(targets).sum().item()
            
            total += targets.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            self.global_step += 1
            
            # Log to wandb
            if self.config['logging']['use_wandb'] and self.global_step % 50 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/lr': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return {'loss': epoch_loss, 'accuracy': epoch_acc}
    
    def validate(self, loader=None, per_species=False) -> Dict[str, float]:
        """Validate model"""
        if loader is None:
            loader = self.val_loader
        
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-species tracking
        species_correct = defaultdict(int)
        species_total = defaultdict(int)
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc='Validation')
            
            for inputs, targets in pbar:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                with autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
                
                # Store for per-species analysis
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update per-species metrics
                for pred, target in zip(predicted.cpu().numpy(), targets.cpu().numpy()):
                    species_total[target] += 1
                    if pred == target:
                        species_correct[target] += 1
                
                pbar.set_postfix({
                    'loss': running_loss / (len(pbar) + 1),
                    'acc': 100. * correct / total
                })
        
        # Calculate metrics
        metrics = {
            'loss': running_loss / len(loader),
            'accuracy': 100. * correct / total
        }
        
        if per_species:
            # Calculate per-species accuracy
            species_acc = {}
            for species_id in species_total:
                acc = 100. * species_correct[species_id] / species_total[species_id]
                species_name = loader.dataset.get_species_name(species_id)
                if isinstance(species_name, dict):
                    species_name = str(species_id)
                species_acc[species_name] = acc
            
            metrics['species_accuracy'] = species_acc
            
            # Log worst performing species
            sorted_species = sorted(species_acc.items(), key=lambda x: x[1])
            self.logger.info("\nWorst performing species:")
            for species, acc in sorted_species[:5]:
                self.logger.info(f"  {species}: {acc:.2f}%")
            
            self.logger.info("\nBest performing species:")
            for species, acc in sorted_species[-5:]:
                self.logger.info(f"  {species}: {acc:.2f}%")
        
        return metrics
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config['checkpoint']['save_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'config': self.config
        }
        
        # Save regular checkpoint
        if (self.epoch + 1) % self.config['checkpoint']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{self.epoch + 1}.pth'
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation accuracy: {self.best_val_accuracy:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_accuracy = checkpoint['best_val_accuracy']
        
        self.logger.info(f"Loaded checkpoint from epoch {self.epoch + 1}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch + 1} - Train Loss: {train_metrics['loss']:.4f}, "
                           f"Train Acc: {train_metrics['accuracy']:.2f}%")
            
            # Validation with per-species metrics every 5 epochs
            val_metrics = self.validate(per_species=(epoch + 1) % 5 == 0)
            self.logger.info(f"Epoch {epoch + 1} - Val Loss: {val_metrics['loss']:.4f}, "
                           f"Val Acc: {val_metrics['accuracy']:.2f}%")
            
            # Update learning rate
            self.scheduler.step()
            
            # Save checkpoint
            is_best = val_metrics['accuracy'] > self.best_val_accuracy
            if is_best:
                self.best_val_accuracy = val_metrics['accuracy']
            
            self.save_checkpoint(is_best=is_best)
            
            # Log to wandb
            if self.config['logging']['use_wandb']:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['loss'],
                    'train/epoch_accuracy': train_metrics['accuracy'],
                    'val/epoch_loss': val_metrics['loss'],
                    'val/epoch_accuracy': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
                # Log per-species metrics if available
                if 'species_accuracy' in val_metrics:
                    for species, acc in val_metrics['species_accuracy'].items():
                        wandb.log({f'val/species/{species}': acc})
            
            # Visualize attention maps every 10 epochs
            if (epoch + 1) % 10 == 0 and self.config['visualization']['save_attention_maps']:
                self.visualize_attention_samples()
        
        self.logger.info("Training completed!")
        
        # Final evaluation on test set
        self.evaluate_test_set()
    
    def visualize_attention_samples(self):
        """Visualize attention maps for sample spectrograms"""
        self.logger.info("Generating attention visualizations...")
        
        viz_dir = Path(self.config['visualization']['output_dir']) / f'epoch_{self.epoch + 1}'
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        visualizer = AttentionVisualizer(self.model, self.device)
        
        # Get a few samples from each class
        samples_per_class = 2
        class_samples = defaultdict(list)
        
        for inputs, targets in self.val_loader:
            for i in range(len(targets)):
                label = targets[i].item()
                if len(class_samples[label]) < samples_per_class:
                    class_samples[label].append(inputs[i:i+1])
            
            if all(len(samples) >= samples_per_class for samples in class_samples.values()):
                break
        
        # Visualize attention for each sample
        for label, samples in class_samples.items():
            species_name = self.val_loader.dataset.get_species_name(label)
            
            for i, sample in enumerate(samples):
                visualizer.save_attention_summary(
                    sample,
                    f"{species_name}_sample_{i}",
                    viz_dir
                )
    
    def evaluate_test_set(self):
        """Final evaluation on test set"""
        self.logger.info("Evaluating on test set...")
        
        # Load best model
        best_checkpoint = Path(self.config['checkpoint']['save_dir']) / 'best_model.pth'
        if best_checkpoint.exists():
            self.load_checkpoint(best_checkpoint)
        
        # Evaluate with detailed metrics
        test_metrics = self.validate(loader=self.test_loader, per_species=True)
        
        self.logger.info(f"\nFinal Test Results:")
        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        
        # Save detailed results
        results = {
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'species_accuracy': test_metrics.get('species_accuracy', {}),
            'model_config': self.config['model'],
            'training_config': self.config['training']
        }
        
        results_path = Path(self.config['checkpoint']['save_dir']) / 'test_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Saved test results to {results_path}")
        
        if self.config['logging']['use_wandb']:
            wandb.log({
                'test/loss': test_metrics['loss'],
                'test/accuracy': test_metrics['accuracy']
            })


def main():
    """Main training function"""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
