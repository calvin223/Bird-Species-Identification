"""
EfficientNet Model for Bird Call Classification
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from efficientnet_pytorch import EfficientNet


class BirdCallDataset(Dataset):
    """Dataset implementation using memory-mapped arrays"""
    
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Load data shape
        shape_file = self.data_dir / f'{split}_features_shape.npy'
        self.shape = np.load(shape_file)
        
        # Memory-map features
        self.features = np.memmap(
            self.data_dir / f'{split}_features.dat',
            dtype='float32',
            mode='r',
            shape=tuple(self.shape)
        )
        
        # Load labels and metadata
        self.labels = np.load(self.data_dir / f'{split}_labels.npy')
        
        with open(self.data_dir / f'{split}_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        print(f"{split} dataset: {len(self.features)} samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        spectrogram = np.array(self.features[idx])
        label = int(self.labels[idx])
        
        spectrogram = torch.from_numpy(spectrogram).float()
        
        if self.transform:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label


class SpecAugment:
    """SpecAugment data augmentation"""
    
    def __init__(self, time_mask_param=20, freq_mask_param=15, n_time_masks=2, n_freq_masks=2):
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
    
    def __call__(self, spectrogram):
        # Apply time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, spectrogram.shape[-1] - t)
            spectrogram[:, :, t0:t0+t] = 0
        
        # Apply frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, spectrogram.shape[-2] - f)
            spectrogram[:, f0:f0+f, :] = 0
        
        return spectrogram


class MixUp:
    """MixUp augmentation"""
    
    def __init__(self, alpha=0.2):
        self.alpha = alpha
    
    def __call__(self, x, y, alpha=None):
        if alpha is None:
            alpha = self.alpha
        
        batch_size = x.size(0)
        lam = np.random.beta(alpha, alpha)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam


class BirdCallEfficientNet(nn.Module):
    """EfficientNet architecture for bird call classification"""
    
    def __init__(self, num_classes=30, model_name='efficientnet-b1', pretrained=True):
        super().__init__()
        
        # Initialize backbone
        if pretrained:
            self.backbone = EfficientNet.from_pretrained(model_name)
        else:
            self.backbone = EfficientNet.from_name(model_name)
        
        # Adapt for single-channel input
        conv_stem = self.backbone._conv_stem
        self.backbone._conv_stem = nn.Conv2d(
            1, conv_stem.out_channels,
            kernel_size=conv_stem.kernel_size,
            stride=conv_stem.stride,
            padding=conv_stem.padding,
            bias=False
        )
        
        # Custom classifier
        num_features = self.backbone._fc.in_features
        self.backbone._fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class BirdCallTrainer:
    """Training pipeline for bird call classification"""
    
    def __init__(self, model, data_dir, output_dir, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load label mapping
        with open(self.data_dir / 'label_to_species_name.json', 'r') as f:
            self.label_to_species = json.load(f)
        
        # Initialize training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        self.best_val_acc = 0
        self.patience_counter = 0
        self.scaler = GradScaler()
    
    def create_data_loaders(self, batch_size=16, num_workers=4):
        """Create data loaders"""
        
        import platform
        if platform.system() == 'Windows':
            num_workers = 0
        
        # Training augmentation
        train_transform = transforms.Compose([
            SpecAugment(time_mask_param=25, freq_mask_param=15),
        ])
        
        # Create datasets
        train_dataset = BirdCallDataset(self.data_dir, 'train', train_transform)
        val_dataset = BirdCallDataset(self.data_dir, 'val')
        test_dataset = BirdCallDataset(self.data_dir, 'test')
        
        # Create loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        self.mixup = MixUp(alpha=0.2)
        
        return len(train_dataset), len(val_dataset), len(test_dataset)
    
    def train_epoch(self, optimizer, criterion, epoch):
        """Execute one training epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Apply MixUp
            if np.random.random() > 0.5:
                inputs, targets_a, targets_b, lam = self.mixup(inputs, targets)
            else:
                targets_a = targets_b = targets
                lam = 1.0
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = self.model(inputs)
                if lam == 1.0:
                    loss = criterion(outputs, targets)
                else:
                    loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(optimizer)
            self.scaler.update()
            
            # Update metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            if lam == 1.0:
                correct += predicted.eq(targets).sum().item()
            else:
                correct += (lam * predicted.eq(targets_a).sum().item() + 
                          (1 - lam) * predicted.eq(targets_b).sum().item())
            
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{running_loss/(batch_idx+1):.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self, criterion):
        """Validate model performance"""
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
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, num_epochs=100, batch_size=16, learning_rate=1e-3, patience=15):
        """Main training loop"""
        
        print(f"\nTraining Configuration:")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Early stopping patience: {patience}")
        
        # Setup data loaders
        train_size, val_size, test_size = self.create_data_loaders(batch_size)
        print(f"\nDataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")
        
        # Initialize loss and optimizer
        criterion = FocalLoss(alpha=1, gamma=2)
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        print("\nStarting training...")
        
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_epoch(optimizer, criterion, epoch)
            val_loss, val_acc = self.validate(criterion)
            scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val: Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.patience_counter = 0
                self.save_checkpoint(epoch, optimizer, val_acc, 'best_model.pth')
                print(f"  ✓ Best model saved!")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            # Periodic checkpoints
            if epoch % 10 == 0:
                self.save_checkpoint(epoch, optimizer, val_acc, f'checkpoint_epoch_{epoch}.pth')
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_val_acc:.2f}%")
        
        self.save_history()
        self.plot_history()
        
        return self.best_val_acc
    
    def save_checkpoint(self, epoch, optimizer, val_acc, filename):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'history': self.history,
            'label_to_species': self.label_to_species
        }
        torch.save(checkpoint, self.output_dir / filename)
    
    def save_history(self):
        with open(self.output_dir / 'training_history.json', 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.history['train_loss'], label='Train')
        ax1.plot(self.history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Curves')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history['train_acc'], label='Train')
        ax2.plot(self.history['val_acc'], label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy Curves')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300)
        plt.close()
    
    def evaluate_test_set(self):
        """Final evaluation on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        checkpoint = torch.load(self.output_dir / 'best_model.pth')
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
        
        # Per-species accuracy
        class_correct = {}
        class_total = {}
        
        for target, pred in zip(all_targets, all_predictions):
            species_name = self.label_to_species[str(target)]['common_name']
            
            if species_name not in class_correct:
                class_correct[species_name] = 0
                class_total[species_name] = 0
            
            class_total[species_name] += 1
            if target == pred:
                class_correct[species_name] += 1
        
        # Save results
        species_accuracies = []
        for species in class_correct:
            acc = 100 * class_correct[species] / class_total[species]
            species_accuracies.append((species, acc, class_total[species]))
        
        species_accuracies.sort(key=lambda x: x[1], reverse=True)
        
        test_results = {
            'test_accuracy': accuracy,
            'per_species_accuracy': {s[0]: s[1] for s in species_accuracies},
            'sample_counts': {s[0]: s[2] for s in species_accuracies}
        }
        
        with open(self.output_dir / 'test_results.json', 'w') as f:
            json.dump(test_results, f, indent=2)
        
        return accuracy


def main():
    # Configuration paths
    data_dir = r"D:\University\Comp702\02_data\bird_calls_30species_memmap"
    output_dir = r"D:\University\Comp702\project\model 1"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    
    print("Bird Call Classification with EfficientNet-B1")
    print("="*80)
    
    # Initialize model
    model = BirdCallEfficientNet(num_classes=30, model_name='efficientnet-b1', pretrained=True)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    trainer = BirdCallTrainer(model, data_dir, output_dir, device)
    
    best_val_acc = trainer.train(
        num_epochs=100,
        batch_size=batch_size,
        learning_rate=1e-3,
        patience=15
    )
    
    test_acc = trainer.evaluate_test_set()
    
    print("\n" + "="*80)
    print(f"Training Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Test Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
