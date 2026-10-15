"""
Efficient data loader for PANNs with memory-mapped arrays and augmentations.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import json
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict


class BirdCallDataset(Dataset):
    """Dataset for bird call mel-spectrograms with PANNs compatibility."""
    
    def __init__(self, root_dir: str, split: str = 'train', 
                 transform=None, mixup_transform=None):
        """
        Args:
            root_dir: Root directory containing the memmap files
            split: 'train', 'val', or 'test'
            transform: Transform to apply to spectrograms
            mixup_transform: MixUp augmentation (applied separately)
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.mixup_transform = mixup_transform
        
        # Load dataset info
        # Try multiple possible info files
        info_files = ['dataset_info.json', 'metadata.json', 'preprocessing_summary.json']
        info_loaded = False
        
        for info_file in info_files:
            info_path = os.path.join(root_dir, info_file)
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.info = json.load(f)
                info_loaded = True
                break
                
        if not info_loaded:
            # Create basic info from available files
            self.info = self._create_info_from_files()
            
        # Get split info
        if 'splits' in self.info:
            self.split_info = self.info['splits'][split]
            self.total_samples = self.split_info['total_samples']
        else:
            # Infer from files
            self.total_samples = self._get_total_samples(split)
            
        # Load memory-mapped arrays
        self._load_memmaps()
        
        # Create label mapping
        if 'species_list' in self.info:
            self.species_list = self.info['species_list']
        else:
            # Try to load from other files
            self._load_species_mapping()
            
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_list)}
        self.idx_to_species = {idx: species for species, idx in self.species_to_idx.items()}
        
        # Calculate samples per species for balanced sampling
        self._calculate_species_distribution()
        
    def _create_info_from_files(self):
        """Create info dict from available files."""
        info = {}
        
        # Try to load split metadata
        split_metadata_path = os.path.join(self.root_dir, f'{self.split}_metadata.json')
        if os.path.exists(split_metadata_path):
            with open(split_metadata_path, 'r') as f:
                metadata = json.load(f)
                info['species_list'] = metadata.get('species_list', [])
                info['spectrogram_shape'] = metadata.get('feature_shape', [1, 128, 259])
                info['num_samples'] = metadata.get('num_samples', 0)
                return info
        
        # Try to load species mapping
        species_mapping_path = os.path.join(self.root_dir, 'species_mapping.json')
        if os.path.exists(species_mapping_path):
            with open(species_mapping_path, 'r') as f:
                species_data = json.load(f)
                if isinstance(species_data, dict):
                    info['species_list'] = list(species_data.keys())
                elif isinstance(species_data, list):
                    info['species_list'] = species_data
                    
        # Try to get shape from shape files
        shape_files = [f for f in os.listdir(self.root_dir) if f.endswith('_shape.npy')]
        if shape_files:
            shape_path = os.path.join(self.root_dir, shape_files[0])
            shape = np.load(shape_path)
            if len(shape) == 4:  # (samples, channels, height, width)
                info['spectrogram_shape'] = list(shape[1:])
            elif len(shape) == 3:  # (samples, height, width)
                info['spectrogram_shape'] = [1] + list(shape[1:])
                
        return info
        
    def _get_total_samples(self, split):
        """Get total samples from shape file or metadata."""
        # Try metadata first
        metadata_path = os.path.join(self.root_dir, f'{split}_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                return metadata.get('num_samples', 0)
                
        # Try shape file
        shape_path = os.path.join(self.root_dir, f'{split}_features_shape.npy')
        if os.path.exists(shape_path):
            shape = np.load(shape_path)
            return shape[0]
        return 0
        
    def _load_species_mapping(self):
        """Load species mapping from various possible files."""
        # Try species_mapping.json
        mapping_files = ['species_mapping.json', 'label_to_species_name.json']
        
        for mapping_file in mapping_files:
            mapping_path = os.path.join(self.root_dir, mapping_file)
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    mapping = json.load(f)
                    if isinstance(mapping, dict):
                        # Check if it's the nested format with common_name/scientific_name
                        first_key = list(mapping.keys())[0]
                        if isinstance(mapping[first_key], dict) and 'common_name' in mapping[first_key]:
                            # Extract common names in order
                            max_idx = max(int(k) for k in mapping.keys())
                            self.species_list = [''] * (max_idx + 1)
                            for k, v in mapping.items():
                                idx = int(k)
                                # Use common name, converting to lowercase with underscores
                                common_name = v['common_name'].lower().replace(' ', '_').replace('-', '_')
                                self.species_list[idx] = common_name
                        else:
                            # Simple key-value mapping
                            max_idx = max(int(k) if k.isdigit() else 0 for k in mapping.keys())
                            self.species_list = [''] * (max_idx + 1)
                            for k, v in mapping.items():
                                if k.isdigit():
                                    self.species_list[int(k)] = v
                    elif isinstance(mapping, list):
                        self.species_list = mapping
                return
                
        # Default list if nothing found
        self.species_list = [f"species_{i}" for i in range(30)]
        
    def _load_memmaps(self):
        """Load memory-mapped arrays."""
        # Try different naming conventions
        spec_names = [f'{self.split}_spectrograms.dat', f'{self.split}_features.dat']
        label_names = [f'{self.split}_labels.dat', f'{self.split}_labels.npy']
        
        # Load spectrograms/features
        spec_loaded = False
        for spec_name in spec_names:
            spec_path = os.path.join(self.root_dir, spec_name)
            if os.path.exists(spec_path):
                # Get shape
                shape_path = os.path.join(self.root_dir, f'{self.split}_features_shape.npy')
                if os.path.exists(shape_path):
                    shape = np.load(shape_path)
                    self.spectrograms = np.memmap(spec_path, dtype='float32', mode='r', shape=tuple(shape))
                else:
                    # Try to infer shape
                    if 'spectrogram_shape' in self.info:
                        spec_shape = (self.total_samples,) + tuple(self.info['spectrogram_shape'])
                        self.spectrograms = np.memmap(spec_path, dtype='float32', mode='r', shape=spec_shape)
                spec_loaded = True
                break
                
        if not spec_loaded:
            raise FileNotFoundError(f"Could not find spectrogram/feature file for {self.split} split")
            
        # Load labels
        label_loaded = False
        for label_name in label_names:
            label_path = os.path.join(self.root_dir, label_name)
            if os.path.exists(label_path):
                if label_name.endswith('.npy'):
                    # Regular numpy array
                    self.labels = np.load(label_path)
                else:
                    # Memory-mapped array
                    self.labels = np.memmap(label_path, dtype='int64', mode='r', shape=(self.total_samples,))
                label_loaded = True
                break
                
        if not label_loaded:
            raise FileNotFoundError(f"Could not find label file for {self.split} split")
        
    def _calculate_species_distribution(self):
        """Calculate distribution of species in the dataset."""
        self.species_counts = defaultdict(int)
        self.species_indices = defaultdict(list)
        
        for idx in range(len(self)):
            label = int(self.labels[idx])
            self.species_counts[label] += 1
            self.species_indices[label].append(idx)
            
        # Calculate sampling weights for balanced training
        total = sum(self.species_counts.values())
        self.sample_weights = torch.zeros(len(self))
        
        for idx in range(len(self)):
            label = int(self.labels[idx])
            # Inverse frequency weighting
            self.sample_weights[idx] = total / (len(self.species_counts) * self.species_counts[label])
            
    def __len__(self):
        return self.total_samples
        
    def __getitem__(self, idx):
        # Load spectrogram and label
        # Use numpy indexing to avoid loading entire array
        spectrogram = np.array(self.spectrograms[idx])  # Force copy to avoid memmap issues
        label = int(self.labels[idx])
        
        # Convert to tensor
        spectrogram = torch.from_numpy(spectrogram).float()
        
        # Ensure correct shape (add channel dimension if needed)
        if spectrogram.dim() == 2:
            spectrogram = spectrogram.unsqueeze(0)
            
        # Apply transforms
        if self.transform is not None:
            result = self.transform(spectrogram)
            # Handle transforms that return tuples
            if isinstance(result, tuple):
                spectrogram = result[0]
            else:
                spectrogram = result
            
        return spectrogram, label
        
    def get_species_name(self, idx):
        """Get species name from index."""
        return self.idx_to_species.get(idx, 'unknown')
        
    def get_balanced_batch(self, batch_size):
        """Get a balanced batch with equal representation of species."""
        samples_per_species = max(1, batch_size // len(self.species_counts))
        indices = []
        
        for species_idx, species_indices in self.species_indices.items():
            # Sample from this species
            n_samples = min(samples_per_species, len(species_indices))
            sampled = random.sample(species_indices, n_samples)
            indices.extend(sampled)
            
        # Shuffle and trim to batch size
        random.shuffle(indices)
        indices = indices[:batch_size]
        
        # Get batch
        batch_specs = []
        batch_labels = []
        
        for idx in indices:
            spec, label = self[idx]
            batch_specs.append(spec)
            batch_labels.append(label)
            
        return torch.stack(batch_specs), torch.tensor(batch_labels)


class BalancedBatchSampler(Sampler):
    """Sampler that ensures balanced representation of classes in each batch."""
    
    def __init__(self, dataset, batch_size, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Group indices by class
        self.class_indices = defaultdict(list)
        for idx in range(len(dataset)):
            label = int(dataset.labels[idx])
            self.class_indices[label].append(idx)
            
        # Calculate batches
        self._prepare_batches()
        
    def _prepare_batches(self):
        """Prepare balanced batches."""
        # Shuffle indices within each class
        for class_idx in self.class_indices:
            random.shuffle(self.class_indices[class_idx])
            
        # Create batches
        self.batches = []
        num_classes = len(self.class_indices)
        samples_per_class = max(1, self.batch_size // num_classes)
        
        # Find minimum class size
        min_class_size = min(len(indices) for indices in self.class_indices.values())
        
        # If we can't get at least one full batch with balanced sampling,
        # fall back to regular sampling
        if min_class_size < samples_per_class:
            print(f"Warning: Minimum class size ({min_class_size}) is less than samples per class ({samples_per_class}).")
            print("Falling back to regular shuffled sampling.")
            # Just create shuffled batches
            all_indices = []
            for indices in self.class_indices.values():
                all_indices.extend(indices)
            random.shuffle(all_indices)
            
            for i in range(0, len(all_indices), self.batch_size):
                batch = all_indices[i:i + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    self.batches.append(batch)
        else:
            # Balanced sampling
            num_batches = min_class_size // samples_per_class
            
            for batch_idx in range(num_batches):
                batch = []
                for class_idx, indices in self.class_indices.items():
                    start_idx = batch_idx * samples_per_class
                    end_idx = start_idx + samples_per_class
                    batch.extend(indices[start_idx:end_idx])
                    
                random.shuffle(batch)
                
                if len(batch) >= self.batch_size:
                    self.batches.append(batch[:self.batch_size])
                elif not self.drop_last and len(batch) > 0:
                    self.batches.append(batch)
                    
        print(f"Created {len(self.batches)} batches for training")
                
    def __iter__(self):
        # Shuffle batches
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch
            
    def __len__(self):
        return len(self.batches)


class MixUpCollate:
    """Custom collate function with MixUp augmentation."""
    
    def __init__(self, mixup_alpha=0.3, mixup_prob=0.8, num_classes=30):
        self.mixup_alpha = mixup_alpha
        self.mixup_prob = mixup_prob
        self.num_classes = num_classes
        
    def __call__(self, batch):
        # Separate spectrograms and labels
        specs, labels = zip(*batch)
        
        # Stack tensors
        specs = torch.stack(specs)
        labels = torch.tensor(labels)
        
        # Apply MixUp if in training mode
        if random.random() < self.mixup_prob:
            batch_size = specs.shape[0]
            
            # Sample lambda
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            
            # Random permutation
            index = torch.randperm(batch_size)
            
            # Mix spectrograms
            mixed_specs = lam * specs + (1 - lam) * specs[index]
            
            # Convert labels to one-hot and mix
            y_onehot = torch.zeros(batch_size, self.num_classes)
            y_onehot.scatter_(1, labels.unsqueeze(1), 1)
            
            mixed_labels = lam * y_onehot + (1 - lam) * y_onehot[index]
            
            return mixed_specs, mixed_labels
            
        return specs, labels


def create_data_loaders(config: Dict, transforms: Dict) -> Dict[str, DataLoader]:
    """Create data loaders for all splits."""
    
    root_dir = config['data']['root_dir']
    batch_size = int(config['training']['batch_size'])
    num_workers = int(config['optimization']['num_workers'])
    pin_memory = bool(config['optimization']['pin_memory'])
    persistent_workers = bool(config['optimization']['persistent_workers'])
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        transform = transforms.get(split, None)
        datasets[split] = BirdCallDataset(
            root_dir=root_dir,
            split=split,
            transform=transform
        )
        
    # Create data loaders
    loaders = {}
    
    # Training loader with balanced sampling and MixUp
    if float(config['augmentation']['mixup']['prob']) > 0:
        collate_fn = MixUpCollate(
            mixup_alpha=float(config['augmentation']['mixup']['alpha']),
            mixup_prob=float(config['augmentation']['mixup']['prob']),
            num_classes=int(config['model']['num_classes'])
        )
    else:
        collate_fn = None
        
    # Use regular sampler instead of balanced sampler for now
    # train_sampler = BalancedBatchSampler(
    #     datasets['train'], 
    #     batch_size=batch_size,
    #     drop_last=True
    # )
    
    loaders['train'] = DataLoader(
        datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_fn
    )
    
    # Validation and test loaders
    for split in ['val', 'test']:
        loaders[split] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0
        )
        
    return loaders, datasets


def calculate_dataset_statistics(dataset: BirdCallDataset) -> Dict:
    """Calculate dataset statistics for monitoring."""
    
    stats = {
        'total_samples': len(dataset),
        'num_species': len(dataset.species_list),
        'species_distribution': {},
        'species_names': dataset.species_list
    }
    
    # Calculate per-species statistics
    for species_idx, count in dataset.species_counts.items():
        species_name = dataset.get_species_name(species_idx)
        stats['species_distribution'][species_name] = {
            'count': count,
            'percentage': (count / len(dataset)) * 100
        }
        
    # Sort by count
    stats['species_distribution'] = dict(
        sorted(stats['species_distribution'].items(), 
               key=lambda x: x[1]['count'], reverse=True)
    )
    
    return stats


if __name__ == "__main__":
    # Test data loader
    import yaml
    from utils.audio_transforms import create_train_transforms, create_val_transforms
    
    # Load config
    with open('config_panns.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Create transforms
    transforms = {
        'train': create_train_transforms(config),
        'val': create_val_transforms(config),
        'test': create_val_transforms(config)
    }
    
    # Create loaders
    loaders, datasets = create_data_loaders(config, transforms)
    
    # Test loading
    print("Testing data loaders...")
    
    for split, loader in loaders.items():
        print(f"\n{split.upper()} loader:")
        print(f"  Batches: {len(loader)}")
        
        # Get one batch
        specs, labels = next(iter(loader))
        print(f"  Batch shape: {specs.shape}")
        print(f"  Labels shape: {labels.shape}")
        
        if split == 'train' and labels.dim() > 1:
            print("  MixUp applied - labels are soft targets")
            
    # Print dataset statistics
    stats = calculate_dataset_statistics(datasets['train'])
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of species: {stats['num_species']}")
    print(f"  Top 5 species:")
    for i, (species, info) in enumerate(stats['species_distribution'].items()):
        if i >= 5:
            break
        print(f"    {species}: {info['count']} ({info['percentage']:.1f}%)")
