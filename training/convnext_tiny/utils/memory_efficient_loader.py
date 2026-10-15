"""
Memory-efficient data loader for memory-mapped bird call spectrograms.
Optimized for Windows and limited GPU memory.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from typing import Optional, Tuple, Dict, List
import warnings
import gc

warnings.filterwarnings("ignore", category=UserWarning)


class MemoryEfficientBirdDataset(Dataset):
    """Memory-efficient dataset using memory-mapped arrays."""
    
    def __init__(self, 
                 data_file: str,
                 labels_file: str,
                 num_samples: int,
                 transform=None,
                 challenging_species_indices: Optional[List[int]] = None,
                 boost_challenging_species: bool = False,
                 boost_factor: float = 1.5):
        """
        Args:
            data_file: Path to memory-mapped data file
            labels_file: Path to memory-mapped labels file
            num_samples: Number of samples in the dataset
            transform: Optional transform to apply
            challenging_species_indices: Indices of challenging species
            boost_challenging_species: Whether to boost challenging species
            boost_factor: Factor to boost challenging species augmentation
        """
        self.data_file = data_file
        self.labels_file = labels_file
        self.num_samples = num_samples
        self.transform = transform
        self.challenging_species_indices = challenging_species_indices or []
        self.boost_challenging_species = boost_challenging_species
        self.boost_factor = boost_factor
        
        # Open memory-mapped arrays
        self._open_memmaps()
        
    def _open_memmaps(self):
        """Open memory-mapped arrays."""
        # Data shape: (num_samples, 1, 128, 259)
        self.data = np.memmap(self.data_file, dtype='float32', mode='r',
                             shape=(self.num_samples, 1, 128, 259))
        
        # Labels are stored as .npy files, not memory-mapped
        self.labels = np.load(self.labels_file)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Load single sample (minimal memory footprint)
        x = torch.from_numpy(self.data[idx].copy()).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        
        # Ensure x has batch dimension for augmentations
        if x.dim() == 3:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply transforms with boosting for challenging species
        if self.transform is not None:
            if self.boost_challenging_species and y.item() in self.challenging_species_indices:
                # Apply stronger augmentation to challenging species
                x = self.transform(x, boost_factor=self.boost_factor)
            else:
                x = self.transform(x)
        
        # Remove batch dimension before returning
        if x.dim() == 4 and x.size(0) == 1:
            x = x.squeeze(0)
        
        return x, y
    
    def __del__(self):
        """Clean up memory-mapped arrays."""
        if hasattr(self, 'data'):
            del self.data
        if hasattr(self, 'labels'):
            del self.labels
        gc.collect()


class MemoryEfficientSampler(torch.utils.data.Sampler):
    """Custom sampler for memory-efficient training."""
    
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0
        
    def __iter__(self):
        # Create indices
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            # Use epoch as seed for reproducibility
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        
        # Create batches
        batch = []
        for idx in indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        
        # Handle last batch
        if len(batch) > 0 and not self.drop_last:
            yield batch
    
    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
    
    def set_epoch(self, epoch):
        self.epoch = epoch


def create_data_loaders(config, transform_train=None, transform_val=None):
    """Create memory-efficient data loaders."""
    
    # Get challenging species indices
    challenging_species_indices = []
    if hasattr(config.dataset, 'challenging_species'):
        for species in config.dataset.challenging_species:
            if species in config.dataset.class_names:
                challenging_species_indices.append(config.dataset.class_names.index(species))
    
    # Create datasets
    train_dataset = MemoryEfficientBirdDataset(
        data_file=os.path.join(config.dataset.root_dir, config.dataset.train_data),
        labels_file=os.path.join(config.dataset.root_dir, config.dataset.train_labels),
        num_samples=111033,  # From config
        transform=transform_train,
        challenging_species_indices=challenging_species_indices,
        boost_challenging_species=config.augmentation.boost_challenging_species,
        boost_factor=config.augmentation.boost_factor
    )
    
    val_dataset = MemoryEfficientBirdDataset(
        data_file=os.path.join(config.dataset.root_dir, config.dataset.val_data),
        labels_file=os.path.join(config.dataset.root_dir, config.dataset.val_labels),
        num_samples=23793,
        transform=transform_val
    )
    
    test_dataset = MemoryEfficientBirdDataset(
        data_file=os.path.join(config.dataset.root_dir, config.dataset.test_data),
        labels_file=os.path.join(config.dataset.root_dir, config.dataset.test_labels),
        num_samples=23793,
        transform=transform_val  # Same as validation
    )
    
    # Create custom samplers
    train_sampler = MemoryEfficientSampler(
        train_dataset, 
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True
    )
    
    # Create data loaders with Windows-specific settings
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.memory.num_workers,  # 0 for Windows
        pin_memory=config.memory.pin_memory,     # False for memmap
        persistent_workers=False                 # Always False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.memory.num_workers,
        pin_memory=config.memory.pin_memory,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.val_batch_size,
        shuffle=False,
        num_workers=config.memory.num_workers,
        pin_memory=config.memory.pin_memory,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader, train_sampler


class DynamicBatchSizeLoader:
    """Wrapper for dynamic batch size adjustment on OOM."""
    
    def __init__(self, loader, min_batch_size=4):
        self.loader = loader
        self.min_batch_size = min_batch_size
        self.current_batch_size = loader.batch_size
        
    def reduce_batch_size(self):
        """Reduce batch size on OOM."""
        if self.current_batch_size > self.min_batch_size:
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size // 2)
            print(f"Reducing batch size to {self.current_batch_size}")
            
            # Recreate loader with new batch size
            dataset = self.loader.dataset
            self.loader = DataLoader(
                dataset,
                batch_size=self.current_batch_size,
                shuffle=False,  # Sampler handles shuffling
                num_workers=0,
                pin_memory=False,
                drop_last=self.loader.drop_last
            )
            return True
        return False
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


def collate_fn_with_mixup(batch, mixup_alpha=0.4, mixup_prob=0.5):
    """Custom collate function with MixUp augmentation."""
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    if np.random.random() < mixup_prob:
        # Apply MixUp
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        images = lam * images + (1 - lam) * images[index]
        
        labels_a, labels_b = labels, labels[index]
        return images, (labels_a, labels_b, lam)
    
    return images, labels


if __name__ == "__main__":
    # Test the data loader
    import yaml
    from types import SimpleNamespace
    
    # Mock config
    config = SimpleNamespace(
        dataset=SimpleNamespace(
            root_dir="../../data/bird_calls_30species_memmap",
            train_data="train_data.dat",
            train_labels="train_labels.dat",
            val_data="val_data.dat", 
            val_labels="val_labels.dat",
            test_data="test_data.dat",
            test_labels="test_labels.dat",
            class_names=["species_" + str(i) for i in range(30)],
            challenging_species=["species_26", "species_25", "species_23"]
        ),
        training=SimpleNamespace(
            batch_size=8,
            val_batch_size=4
        ),
        memory=SimpleNamespace(
            num_workers=0,
            pin_memory=False
        ),
        augmentation=SimpleNamespace(
            boost_challenging_species=True,
            boost_factor=1.5
        )
    )
    
    # Create loaders
    train_loader, val_loader, test_loader, _ = create_data_loaders(config)
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")
    
    # Test one batch
    for batch_idx, (data, target) in enumerate(train_loader):
        print(f"Batch {batch_idx}: data shape {data.shape}, target shape {target.shape}")
        break
