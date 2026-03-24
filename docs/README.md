# Bird Species Identification System

## Overview
6-model ensemble system for identifying 30 UK bird species from audio recordings.
- **Accuracy**: 97.28% on test set
- **Models**: EfficientNet-B1, ResNet-50 CBAM, DenseNet-121, AST, ConvNeXt-Tiny, PANNs CNN14
- **Processing**: 3-second segments with 50% overlap

## Requirements
- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)
- Libraries: librosa, numpy, torchaudio (optional)

## Quick Start

### 1. Verify System
```bash
python verify.py
```

### 2. Identify Bird Species
```bash
python bird_identifier.py <audio_file>
```

Example:
```bash
python bird_identifier.py recording.wav
```

## Preprocessing Pipeline
1. Load audio at 44,100 Hz
2. Convert to mono
3. Split into 3-second segments (50% overlap)
4. Apply pre-emphasis filter (coef=0.97)
5. Generate mel-spectrograms (128 mels, 2048 FFT)
6. Convert to dB scale (top_db=80)
7. Normalize to zero mean and unit variance

## Model Performance

| Model | Test Accuracy | Parameters |
|-------|--------------|------------|
| EfficientNet-B1 | 95.77% | 7.2M |
| ResNet-50 CBAM | 98.15% | 28.2M |
| DenseNet-121 | 86.33% | 8.0M |
| AST | 85.60% | 10.9M |
| ConvNeXt-Tiny | 96.52% | 27.8M |
| PANNs CNN14 | 88.59% | 88.1M |
| **Ensemble** | **97.28%** | **170.2M** |

## Supported Species
30 UK bird species including:
- Barn Owl, Tawny Owl
- Blue Tit, Great Tit, Coal Tit, Long-tailed Tit
- European Robin, Blackbird, Dunnock
- Chaffinch, Bullfinch, Greenfinch
- And 18 more species

## Output Interpretation

### High Confidence (>70%)
Strong identification with model consensus. Reliable result.

### Medium Confidence (35-70%)
Probable identification. Check recording quality if unexpected.

### Low Confidence (<35%)
Likely out-of-scope species or poor recording quality.

## Files
- `bird_identifier.py` - Main identification script
- `verify.py` - System verification
- `ensemble_model.py` - Model ensemble implementation
- `preprocessing_fixed.py` - Audio preprocessing
- `configs/ensemble_config.yaml` - Configuration

## Troubleshooting

### Low confidence on known species
- Ensure clear bird vocalizations
- Minimize background noise
- Use recordings >3 seconds

### Models disagree
- Likely out-of-scope species
- Multiple birds vocalizing
- Poor audio quality

### No valid segments
- Audio too quiet
- Recording too short (<3 seconds)
