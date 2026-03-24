"""
Preprocessing Pipeline for Bird Call Dataset
Creates memory-mapped arrays for efficient training of large models
"""

import os
import numpy as np
import librosa
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Audio processing parameters
SAMPLE_RATE = 44100
DURATION = 3.0  # seconds
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
F_MIN = 300
F_MAX = 15000

# Dataset parameters
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_STATE = 42
MIN_FILES_PER_SPECIES = 50  # Minimum recordings required per species


def load_audio(file_path, sr=SAMPLE_RATE, duration=DURATION):
    """
    Load and preprocess audio file.
    
    Args:
        file_path: Path to audio file
        sr: Sample rate
        duration: Duration to load in seconds
        
    Returns:
        Audio signal array
    """
    try:
        # Load audio with fixed duration
        y, _ = librosa.load(file_path, sr=sr, duration=duration, mono=True)
        
        # Pad if shorter than expected duration
        expected_length = int(sr * duration)
        if len(y) < expected_length:
            y = np.pad(y, (0, expected_length - len(y)), mode='constant')
        else:
            y = y[:expected_length]
        
        return y
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def compute_mel_spectrogram(audio, sr=SAMPLE_RATE):
    """
    Compute mel-spectrogram from audio signal.
    
    Args:
        audio: Audio signal array
        sr: Sample rate
        
    Returns:
        Mel-spectrogram in dB scale
    """
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        fmin=F_MIN,
        fmax=F_MAX
    )
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [-1, 1]
    mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / mel_spec_db.std()
    
    return mel_spec_db


def select_30_species(data_dir, output_dir):
    """
    Select 30 bird species with most recordings for balanced dataset.
    
    Args:
        data_dir: Directory containing bird recordings organized by species
        output_dir: Directory to save selection results
        
    Returns:
        Dictionary mapping species names to file paths
    """
    print("Analyzing available species...")
    
    species_files = {}
    data_path = Path(data_dir)
    
    # Collect all species and their files
    for species_dir in data_path.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            audio_files = []
            
            # Find all audio files
            for ext in ['*.wav', '*.mp3', '*.flac', '*.ogg']:
                audio_files.extend(species_dir.glob(ext))
            
            if len(audio_files) >= MIN_FILES_PER_SPECIES:
                species_files[species_name] = audio_files
                print(f"  {species_name}: {len(audio_files)} recordings")
    
    # Sort by number of recordings and select top 30
    sorted_species = sorted(species_files.items(), key=lambda x: len(x[1]), reverse=True)
    selected_species = dict(sorted_species[:30])
    
    print(f"\nSelected {len(selected_species)} species with most recordings")
    
    # Create mapping for labels
    species_to_label = {name: idx for idx, name in enumerate(sorted(selected_species.keys()))}
    
    # Save selection info
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    selection_info = {
        'selected_species': list(selected_species.keys()),
        'species_to_label': species_to_label,
        'num_files_per_species': {name: len(files) for name, files in selected_species.items()},
        'total_files': sum(len(files) for files in selected_species.values())
    }
    
    with open(output_path / 'species_selection_info.json', 'w') as f:
        json.dump(selection_info, f, indent=2)
    
    print(f"Total files to process: {selection_info['total_files']}")
    
    return selected_species, species_to_label


def preprocess_dataset(selected_species, species_to_label, output_dir):
    """
    Preprocess audio files and create memory-mapped arrays.
    
    Args:
        selected_species: Dictionary of species names to file paths
        species_to_label: Dictionary mapping species names to labels
        output_dir: Directory to save preprocessed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nProcessing audio files...")
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    # Process each species
    for species_name, file_paths in tqdm(selected_species.items(), desc="Species"):
        label = species_to_label[species_name]
        
        for file_path in tqdm(file_paths[:200], desc=f"  {species_name}", leave=False):  # Limit files per species
            # Load audio
            audio = load_audio(file_path)
            if audio is None:
                continue
            
            # Compute mel-spectrogram
            mel_spec = compute_mel_spectrogram(audio)
            
            # Add channel dimension
            mel_spec = np.expand_dims(mel_spec, axis=0)
            
            # Store
            all_features.append(mel_spec)
            all_labels.append(label)
            all_metadata.append({
                'species': species_name,
                'file_path': str(file_path),
                'label': label
            })
    
    # Convert to numpy arrays
    all_features = np.array(all_features, dtype='float32')
    all_labels = np.array(all_labels, dtype='int64')
    
    print(f"\nTotal samples processed: {len(all_features)}")
    print(f"Feature shape: {all_features[0].shape}")
    
    # Split into train/val/test
    print("\nSplitting dataset...")
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test, meta_temp, meta_test = train_test_split(
        all_features, all_labels, all_metadata,
        test_size=TEST_RATIO,
        stratify=all_labels,
        random_state=RANDOM_STATE
    )
    
    # Second split: train vs val
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X_temp, y_temp, meta_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=RANDOM_STATE
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Save as memory-mapped arrays
    print("\nSaving memory-mapped arrays...")
    
    for split_name, features, labels, metadata in [
        ('train', X_train, y_train, meta_train),
        ('val', X_val, y_val, meta_val),
        ('test', X_test, y_test, meta_test)
    ]:
        # Save features as memory-mapped array
        shape = features.shape
        memmap_path = output_path / f'{split_name}_features.dat'
        memmap_array = np.memmap(
            memmap_path,
            dtype='float32',
            mode='w+',
            shape=shape
        )
        memmap_array[:] = features[:]
        memmap_array.flush()
        
        # Save shape information
        np.save(output_path / f'{split_name}_features_shape.npy', shape)
        
        # Save labels
        np.save(output_path / f'{split_name}_labels.npy', labels)
        
        # Save metadata
        with open(output_path / f'{split_name}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  {split_name}: Saved {len(features)} samples")
    
    # Create label to species name mapping
    label_to_species = {}
    for species_name, label in species_to_label.items():
        label_to_species[str(label)] = {
            'common_name': species_name,
            'scientific_name': get_scientific_name(species_name)
        }
    
    with open(output_path / 'label_to_species_name.json', 'w') as f:
        json.dump(label_to_species, f, indent=2)
    
    print("\nPreprocessing complete!")
    print(f"Data saved to: {output_path}")


def get_scientific_name(common_name):
    """
    Get scientific name for a bird species.
    This is a simplified mapping - in production, would use a proper taxonomy database.
    
    Args:
        common_name: Common name of the species
        
    Returns:
        Scientific name
    """
    scientific_names = {
        "Barn Owl": "Tyto alba",
        "Black-headed Gull": "Chroicocephalus ridibundus",
        "Blackcap": "Sylvia atricapilla",
        "Blue Tit": "Cyanistes caeruleus",
        "Bullfinch": "Pyrrhula pyrrhula",
        "Chaffinch": "Fringilla coelebs",
        "Chiffchaff": "Phylloscopus collybita",
        "Coal Tit": "Periparus ater",
        "Common Blackbird": "Turdus merula",
        "Coot": "Fulica atra",
        "Dunnock": "Prunella modularis",
        "Eurasian Magpie": "Pica pica",
        "Eurasian Wren": "Troglodytes troglodytes",
        "European Greenfinch": "Chloris chloris",
        "European Robin": "Erithacus rubecula",
        "Fieldfare": "Turdus pilaris",
        "Goldcrest": "Regulus regulus",
        "Great Spotted Woodpecker": "Dendrocopos major",
        "Great Tit": "Parus major",
        "House Sparrow": "Passer domesticus",
        "Jackdaw": "Corvus monedula",
        "Long-tailed Tit": "Aegithalos caudatus",
        "Mallard": "Anas platyrhynchos",
        "Moorhen": "Gallinula chloropus",
        "Nuthatch": "Sitta europaea",
        "Pied Wagtail": "Motacilla alba",
        "Starling": "Sturnus vulgaris",
        "Swallow": "Hirundo rustica",
        "Tawny Owl": "Strix aluco",
        "Water Rail": "Rallus aquaticus"
    }
    
    return scientific_names.get(common_name, "Unknown")


def verify_dataset(output_dir):
    """
    Verify the preprocessed dataset integrity.
    
    Args:
        output_dir: Directory containing preprocessed data
    """
    output_path = Path(output_dir)
    
    print("\nVerifying dataset...")
    
    for split in ['train', 'val', 'test']:
        # Load shape
        shape = np.load(output_path / f'{split}_features_shape.npy')
        
        # Load memory-mapped array
        features = np.memmap(
            output_path / f'{split}_features.dat',
            dtype='float32',
            mode='r',
            shape=tuple(shape)
        )
        
        # Load labels
        labels = np.load(output_path / f'{split}_labels.npy')
        
        # Load metadata
        with open(output_path / f'{split}_metadata.json', 'r') as f:
            metadata = json.load(f)
        
        print(f"{split}:")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Metadata entries: {len(metadata)}")
        print(f"  Unique labels: {len(np.unique(labels))}")
        print(f"  Label range: [{labels.min()}, {labels.max()}]")
        
        # Verify consistency
        assert len(features) == len(labels) == len(metadata), f"Inconsistent lengths in {split}"
        assert features.shape[1:] == (1, N_MELS, 259), f"Unexpected feature shape in {split}"
    
    print("\n✓ Dataset verification passed!")


def main():
    """Main preprocessing pipeline"""
    # Configuration
    raw_data_dir = r"D:\University\Comp702\01_raw_data\bird_recordings"
    output_dir = r"D:\University\Comp702\02_data\bird_calls_30species_memmap"
    
    print("Bird Call Dataset Preprocessing Pipeline")
    print("=" * 60)
    print(f"Raw data directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"\nAudio parameters:")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Duration: {DURATION} seconds")
    print(f"  Mel bands: {N_MELS}")
    print(f"  Frequency range: {F_MIN}-{F_MAX} Hz")
    
    # Select 30 species with most recordings
    selected_species, species_to_label = select_30_species(raw_data_dir, output_dir)
    
    # Preprocess and create memory-mapped arrays
    preprocess_dataset(selected_species, species_to_label, output_dir)
    
    # Verify dataset integrity
    verify_dataset(output_dir)
    
    print("\nPreprocessing pipeline complete!")


if __name__ == "__main__":
    main()
