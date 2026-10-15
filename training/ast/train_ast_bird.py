"""
Fast training script with optimizations

Trains AST_Bird (models/ast_bird.py), the Audio Spectrogram Transformer used in
the ensemble. The best weights are saved to fast_model_best.pth.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.amp import GradScaler
import time
from pathlib import Path
import json
from tqdm import tqdm

class FastBirdDataset(Dataset):
    """Optimized dataset with memory mapping"""
    def __init__(self, data_path, split='train'):
        self.data_path = Path(data_path)
        
        # Load metadata
        with open(self.data_path / f'{split}_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Memory map data for efficient loading
        shape = (self.metadata['num_samples'], *self.metadata['feature_shape'])
        self.spectrograms = np.memmap(
            self.data_path / f'{split}_features.dat',
            dtype='float32',
            mode='r',
            shape=shape
        )
        
        # Load labels
        self.labels = np.load(self.data_path / f'{split}_labels.npy')
        print(f"Loaded {split} dataset: {len(self.labels)} samples")
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Load from memory-mapped file
        spec = torch.from_numpy(self.spectrograms[idx].copy()).float()
        label = int(self.labels[idx])
        return spec, label


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Create datasets (path is relative to this folder, so run the script from here)
    print("\nCreating datasets...")
    train_dataset = FastBirdDataset(
        '../../data/bird_calls_30species_memmap',
        split='train'
    )

    val_dataset = FastBirdDataset(
        '../../data/bird_calls_30species_memmap',
        split='val'
    )
    
    # Larger batch size for faster training
    batch_size = 128  # Doubled from 64
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0,  # Keep at 0 for Windows
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,  # Even larger for validation
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    print(f"Train batches: {len(train_loader)} ({batch_size} samples/batch)")
    print(f"Val batches: {len(val_loader)}")
    
    # Create smaller model for faster training
    from models.ast_bird import create_ast_bird
    
    print("\nCreating model...")
    model = create_ast_bird(
        num_classes=30,
        pretrained=None,
        embed_dim=384,  # Smaller model: 384 instead of 768
        depth=6,        # Fewer layers: 6 instead of 12
        num_heads=6,    # Fewer heads: 6 instead of 12
        drop_rate=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1
    )
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.1f}M (reduced from 85.5M)")
    
    # Optimized training settings
    criterion = nn.CrossEntropyLoss()
    
    # Lower learning rate for more stable training
    base_lr = 0.0001
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.05)
    
    # Use cosine annealing with warmup
    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = 50 * len(train_loader)
    
    def get_lr(step):
        if step < warmup_steps:
            # Linear warmup
            return base_lr * (step / warmup_steps)
        else:
            # Cosine annealing after warmup
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return base_lr * (1 + np.cos(np.pi * progress)) / 2
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step) / base_lr)
    
    # Add gradient clipping for stability
    max_grad_norm = 1.0
    
    # Mixed precision training
    scaler = GradScaler('cuda')
    
    # Training loop
    print("\nStarting fast training...")
    print("="*60)
    
    global_step = 0  # Track total steps for scheduler
    
    for epoch in range(50):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        epoch_start = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
        for batch_idx, (specs, labels) in enumerate(pbar):
            specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Mixed precision forward pass
            with autocast('cuda'):
                logits, _ = model(specs)
                loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)  # More efficient
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1
            
            # Metrics
            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == labels).sum().item()
            train_total += labels.size(0)
            
            # Update progress bar every 10 batches
            if batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{(train_correct/train_total)*100:.2f}%',
                    'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
                })
        
        # Quick validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for specs, labels in tqdm(val_loader, desc='Validation', leave=False):
                specs, labels = specs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                
                with autocast('cuda'):
                    logits, _ = model(specs)
                
                pred = logits.argmax(dim=1)
                val_correct += (pred == labels).sum().item()
                val_total += labels.size(0)
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        train_acc = (train_correct / train_total) * 100
        val_acc = (val_correct / val_total) * 100
        
        print(f"\nEpoch {epoch+1}/50 - Time: {epoch_time:.1f}s")
        print(f"  Train Acc: {train_acc:.2f}%")
        print(f"  Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print("-"*60)
        
        # Early stopping
        if val_acc > 85.0:
            print(f"\nReached {val_acc:.2f}% validation accuracy! Stopping early.")
            torch.save(model.state_dict(), 'fast_model_best.pth')
            break
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'checkpoint_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main()
