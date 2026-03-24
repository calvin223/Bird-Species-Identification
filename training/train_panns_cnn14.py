"""
Fast training script for PANNs CNN14 - simplified version for debugging
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import yaml
import argparse
from datetime import datetime

# Import custom modules
from models.panns_cnn14 import BirdCallCNN14
from utils.data_loader_panns import create_data_loaders
from utils.audio_transforms import create_train_transforms, create_val_transforms


class SimplifiedTrainer:
    def __init__(self, config_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create data loaders
        print("Creating data loaders...")
        transforms = {
            'train': create_val_transforms(self.config),  # Use simple transforms
            'val': create_val_transforms(self.config),
            'test': create_val_transforms(self.config)
        }
        
        self.loaders, self.datasets = create_data_loaders(self.config, transforms)
        print(f"Training samples: {len(self.datasets['train'])}")
        
        # Create model
        print("Creating model...")
        # Load pretrained model
        from models.pretrained_init import create_pretrained_model
        checkpoint_url = self.config['model'].get('checkpoint_url', 
            'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1')
        
        self.model = create_pretrained_model(
            BirdCallCNN14,
            num_classes=self.config['model']['num_classes'],
            checkpoint_url=checkpoint_url,
            device=self.device
        )
        
        # Freeze all layers except the final classifier
        for name, param in self.model.named_parameters():
            if 'fc_audioset' not in name:  # Only train the final layer
                param.requires_grad = False
                
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Simple loss and optimizer - only optimize trainable params
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=1e-3
        )
        
        print(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Enable mixed precision
        self.scaler = GradScaler()
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.loaders['train'], desc='Training')
        
        for batch_idx, (specs, labels) in enumerate(pbar):
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(specs)
                
                # Get clipwise output
                if isinstance(outputs, dict):
                    logits = outputs['clipwise_output']
                else:
                    logits = outputs
                    
                loss = self.criterion(logits, labels)
            
            # Backward pass with mixed precision
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
            
            # Quick test - train for only 100 batches
            if batch_idx >= 100:
                break
                
        return avg_loss, acc
        
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.loaders['val'], desc='Validation')
        
        for batch_idx, (specs, labels) in enumerate(pbar):
            specs = specs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(specs)
            
            if isinstance(outputs, dict):
                logits = outputs['clipwise_output']
            else:
                logits = outputs
                
            loss = self.criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            acc = 100. * correct / total
            avg_loss = total_loss / (batch_idx + 1)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.2f}%'})
            
            # Quick validation - only 50 batches
            if batch_idx >= 50:
                break
                
        return avg_loss, acc
        
    def train(self):
        print("\nStarting fast training...")
        
        for epoch in range(5):  # Only 5 epochs for testing
            print(f"\nEpoch {epoch+1}/5")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validate
            val_loss, val_acc = self.validate()
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
        print("\nFast training completed!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_panns.yaml')
    args = parser.parse_args()
    
    trainer = SimplifiedTrainer(args.config)
    trainer.train()


if __name__ == "__main__":
    main()
