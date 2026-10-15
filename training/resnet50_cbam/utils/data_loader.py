import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import os
from typing import Dict, Tuple, Optional, List
import logging
from collections import Counter


class BirdCallDataset(Dataset):
    """Dataset for bird call spectrograms using memory-mapped arrays"""
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 transform=None,
                 cache_size: int = 1000,
                 use_caching: bool = True):
        """
        Args:
            data_dir: Root directory containing the dataset
            split: One of 'train', 'val', or 'test'
            transform: Optional transform to apply to spectrograms
            cache_size: Number of samples to cache in memory
            use_caching: Whether to use LRU caching for frequently accessed samples
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        
        # Load metadata
        with open(os.path.join(data_dir, f'{split}_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
        # Load labels
        self.labels = np.load(os.path.join(data_dir, f'{split}_labels.npy'))
        
        # Load species mapping
        with open(os.path.join(data_dir, 'species_mapping.json'), 'r') as f:
            self.species_mapping = json.load(f)
        
        with open(os.path.join(data_dir, 'label_to_species_name.json'), 'r') as f:
            self.label_to_species = json.load(f)
        
        # Setup memory-mapped array
        self.features_path = os.path.join(data_dir, f'{split}_features.dat')
        self.n_samples = self.metadata['num_samples']  # Changed from 'n_samples'
        self.feature_shape = tuple(self.metadata['feature_shape'])  # (1, 128, 259)
        
        # Memory map the features
        self.features_mmap = np.memmap(
            self.features_path,
            dtype='float32',
            mode='r',
            shape=(self.n_samples, *self.feature_shape)
        )
        
        # Setup caching
        self.cache_size = cache_size
        if use_caching:
            self._cache = {}
        else:
            self._cache = None
        # Calculate class weights for balanced sampling
        self._calculate_class_weights()
        
        logging.info(f"Loaded {split} dataset with {self.n_samples} samples")
        logging.info(f"Feature shape: {self.feature_shape}")
        logging.info(f"Number of classes: {len(self.species_mapping)}")
    
    def _calculate_class_weights(self):
        """Calculate class weights for balanced sampling"""
        label_counts = Counter(self.labels)
        total_samples = len(self.labels)
        
        # Calculate weight for each class
        self.class_weights = {}
        for label, count in label_counts.items():
            self.class_weights[label] = total_samples / (len(label_counts) * count)
        
        # Create sample weights
        self.sample_weights = [self.class_weights[label] for label in self.labels]
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        if self._cache is not None:
            if idx in self._cache:
                return self._cache[idx]
            else:
                result = self._get_item_impl(idx)
                if len(self._cache) < self.cache_size:
                    self._cache[idx] = result
                return result
        return self._get_item_impl(idx)
    
    def _get_item_impl(self, idx):
        """Actual implementation of __getitem__"""
        # Get spectrogram from memory-mapped array
        spectrogram = self.features_mmap[idx].copy()  # Copy to avoid mmap issues
        
        # Data already has shape (1, 128, 259), no need to add channel dimension
        
        # Get label
        label = int(self.labels[idx])
        
        # Convert to tensor
        spectrogram = torch.from_numpy(spectrogram).float()
        
        # Apply transforms if any
        if self.transform is not None:
            spectrogram = self.transform(spectrogram)
        
        return spectrogram, label
    
    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler"""
        return self.sample_weights
    
    def get_class_weights(self):
        """Get class weights as tensor for loss function"""
        weights = [self.class_weights[i] for i in range(len(self.class_weights))]
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_species_name(self, label: int) -> str:
        """Get species name from label"""
        return self.label_to_species[str(label)]
    
    def get_label_distribution(self) -> Dict[str, int]:
        """Get distribution of labels in the dataset"""
        distribution = {}
        for label in self.labels:
            species = self.get_species_name(label)
            distribution[species] = distribution.get(species, 0) + 1
        return distribution


class BirdCallDataLoader:
    """Factory class for creating data loaders"""
    
    def __init__(self, 
                 data_dir: str,
                 batch_size: int = 64,
                 num_workers: int = 8,
                 pin_memory: bool = True,
                 use_weighted_sampling: bool = True,
                 augmentation_fn=None):
        """
        Args:
            data_dir: Root directory containing the dataset
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading
            pin_memory: Whether to pin memory for faster GPU transfer
            use_weighted_sampling: Whether to use weighted random sampling
            augmentation_fn: Augmentation function to apply during training
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.use_weighted_sampling = use_weighted_sampling
        self.augmentation_fn = augmentation_fn
    
    def get_train_loader(self) -> DataLoader:
        """Create training data loader with augmentation and weighted sampling"""
        dataset = BirdCallDataset(
            self.data_dir, 
            split='train',
            transform=self.augmentation_fn,
            use_caching=True
        )
        
        if self.use_weighted_sampling:
            sampler = WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True
            )
            shuffle = False
        else:
            sampler = None
            shuffle = True
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def get_val_loader(self) -> DataLoader:
        """Create validation data loader"""
        dataset = BirdCallDataset(
            self.data_dir,
            split='val',
            transform=None,  # No augmentation for validation
            use_caching=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,  # Can use larger batch for validation
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def get_test_loader(self) -> DataLoader:
        """Create test data loader"""
        dataset = BirdCallDataset(
            self.data_dir,
            split='test',
            transform=None,
            use_caching=True
        )
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=2 if self.num_workers > 0 else None
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Get class weights for loss function"""
        train_dataset = BirdCallDataset(
            self.data_dir,
            split='train',
            use_caching=False
        )
        return train_dataset.get_class_weights()


def collate_fn_mixup(batch: List[Tuple[torch.Tensor, int]], 
                     alpha: float = 0.3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Custom collate function for MixUp augmentation"""
    spectrograms, labels = zip(*batch)
    spectrograms = torch.stack(spectrograms)
    labels = torch.tensor(labels)
    
    if np.random.random() < 0.5:  # Apply MixUp 50% of the time
        batch_size = spectrograms.size(0)
        indices = torch.randperm(batch_size)
        
        lam = np.random.beta(alpha, alpha)
        mixed_spectrograms = lam * spectrograms + (1 - lam) * spectrograms[indices]
        
        return mixed_spectrograms, labels, labels[indices], torch.tensor(lam)
    
    return spectrograms, labels, labels, torch.tensor(1.0)


def create_data_loaders(config: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create all data loaders from config"""
    from .augmentations import get_train_transforms
    
    augmentation_fn = get_train_transforms(config['augmentation'])
    
    data_loader_factory = BirdCallDataLoader(
        data_dir=config['data']['data_dir'],
        batch_size=config['training']['batch_size'],
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        use_weighted_sampling=config['data']['use_weighted_sampling'],
        augmentation_fn=augmentation_fn
    )
    
    train_loader = data_loader_factory.get_train_loader()
    val_loader = data_loader_factory.get_val_loader()
    test_loader = data_loader_factory.get_test_loader()
    
    return train_loader, val_loader, test_loader