"""
Audio Spectrogram Transformer (AST) for Bird Call Classification
Transformer-based architecture for audio pattern recognition
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# AST model implementation
from transformers import ASTModel, ASTConfig


class BirdCallDataset(Dataset):
    """Dataset for bird call spectrograms"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load data shapes
        shape_file = self.data_dir / f'{split}_features_shape.npy'
        self.shape = np.load(shape_file)
        
        # Memory-map the features for efficient loading
        self.features = np.memmap(
            self.data_dir / f'{split}_features.dat',
            dtype='float32',
            mode='r',
            shape=tuple(self.shape)
        )
        
        # Load labels
        self.labels = np.load(self.data_dir / f'{split}_labels.npy')
        
        # Load metadata
        with open(self.data_dir / f'{split}_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"Loaded {split} dataset: {len(self.features)} samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spectrogram = np.array(self.features[idx])
        label = int(self.labels[idx])
        
        # Apply transforms if any
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return torch.from_numpy(spectrogram).float(), label


class BirdCallAST(nn.Module):
    """Audio Spectrogram Transformer for bird call classification"""
    
    def __init__(self, num_classes=30, pretrained=True):
        super().__init__()
        
        # Initialize AST configuration
        if pretrained:
            # Load pretrained AST model
            self.ast = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
            self.config = self.ast.config
        else:
            # Create AST from scratch with custom config
            self.config = ASTConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_length=1024,
                frequency_stride=10,
                time_stride=10,
                num_mel_bins=128
            )
            self.ast = ASTModel(self.config)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of classifier head"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
    
    def forward(self, x):
        # Ensure input is in correct shape for AST
        if x.dim() == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Get AST outputs
        outputs = self.ast(x)
        
        # Use the pooled output (CLS token)
        pooled_output = outputs.last_hidden_state[:, 0]
        
        # Classification
        logits = self.classifier(pooled_output)
        
        return logits


class ASTTrainer:
    """Training pipeline for AST model"""
    
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.patience_counter = 0
        
        # Mixed precision scaler
        self.scaler = GradScaler()
    
    def create_data_loaders(self, data_dir, batch_size=8, num_workers=0):
        """Create data loaders"""
        # Load datasets
        train_dataset = BirdCallDataset(Path(data_dir), 'train')
        val_dataset = BirdCallDataset(Path(data_dir), 'val')
        test_dataset = BirdCallDataset(Path(data_dir), 'test')
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        # Load label mapping
        with open(Path(data_dir) / 'label_to_species_name.json', 'r') as f:
            self.label_to_species = json.load(f)
        
        return len(train_dataset), len(val_dataset), len(test_dataset)
    
    def train_epoch(self, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': running_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc='Validation'):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        val_loss = running_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        
        return val_loss, val_acc
    
    def train(self, data_dir, output_dir, num_epochs=50, batch_size=8, 
              learning_rate=5e-5, patience=10):
        """Complete training pipeline"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Device: {self.device}")
        
        # Create data loaders
        train_size, val_size, test_size = self.create_data_loaders(data_dir, batch_size)
        print(f"\nDataset sizes:")
        print(f"  Train: {train_size}")
        print(f"  Val: {val_size}")
        print(f"  Test: {test_size}")
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer - using smaller learning rate for transformer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        print("\nStarting training...")
        print("-" * 60)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate(criterion)
            
            # Update scheduler
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, optimizer, val_acc, 
                                   output_dir / 'best_model.pth')
                print(f"  ✓ New best model saved! Val Acc: {val_acc:.2f}%")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered after {epoch} epochs")
                break
        
        print("\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        
        # Save training history
        self.save_history(output_dir)
        
        # Plot training curves
        self.plot_history(output_dir)
        
        # Evaluate on test set
        self.evaluate_test_set(output_dir)
    
    def save_checkpoint(self, epoch, optimizer, val_acc, filepath):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        torch.save(checkpoint, filepath)
    
    def save_history(self, output_dir):
        """Save training history"""
        with open(output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self, output_dir):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_curves.png', dpi=300)
        plt.close()
    
    def evaluate_test_set(self, output_dir):
        """Evaluate model on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        checkpoint = torch.load(output_dir / 'best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.test_loader, desc='Testing'):
                inputs = inputs.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.numpy())
        
        # Calculate accuracy
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        accuracy = 100 * np.mean(all_predictions == all_targets)
        print(f"\nTest Accuracy: {accuracy:.2f}%")
        
        # Generate confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        species_names = [self.label_to_species[str(i)]['common_name'] 
                        for i in range(len(self.label_to_species))]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=species_names, yticklabels=species_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        
        # Save test results
        test_results = {
            'test_accuracy': accuracy,
            'confusion_matrix': cm.tolist()
        }
        
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return accuracy


def main():
    # Configuration
    data_dir = r"D:\University\Comp702\02_data\bird_calls_30species_memmap"
    output_dir = r"D:\University\Comp702\project\model 4"
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Audio Spectrogram Transformer for Bird Call Classification")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    
    # Create model
    model = BirdCallAST(num_classes=30, pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: Audio Spectrogram Transformer")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = ASTTrainer(model, device)
    
    # Train model
    trainer.train(
        data_dir=data_dir,
        output_dir=output_dir,
        num_epochs=50,
        batch_size=8,
        learning_rate=5e-5,
        patience=10
    )


if __name__ == "__main__":
    main()
