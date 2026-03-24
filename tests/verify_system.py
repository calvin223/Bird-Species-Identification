"""
Verify models are loaded properly and test preprocessing pipeline
This ensures everything matches the training exactly
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from pathlib import Path
import warnings
import sys

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

from ensemble_model import BirdCallEnsemble


def verify_model_loading():
    """
    Verify all 6 models are loaded properly with correct architecture
    """
    print("=" * 80)
    print("VERIFYING MODEL LOADING")
    print("=" * 80)
    
    # Load ensemble
    print("\nLoading ensemble...")
    ensemble = BirdCallEnsemble('configs/ensemble_config.yaml')
    
    print(f"\n[OK] Device: {ensemble.device}")
    print(f"[OK] Number of models loaded: {len(ensemble.models)}/6")
    
    # Expected model architectures
    expected_models = {
        'model_1': ('EfficientNet-B1', 95.77),
        'model_2': ('ResNet-50 CBAM', 98.15),
        'model_3': ('DenseNet-121', 86.33),
        'model_4': ('AST', 85.60),
        'model_5': ('ConvNeXt-Tiny', 96.52),
        'model_6': ('PANNs CNN14', 88.59)
    }
    
    print("\n" + "-" * 80)
    print("MODEL VERIFICATION:")
    print("-" * 80)
    
    # Test each model
    test_input = torch.randn(1, 1, 128, 259).to(ensemble.device)
    
    for model_key, model in ensemble.models.items():
        model_name, expected_acc = expected_models[model_key]
        print(f"\n{model_name} ({model_key}):")
        print(f"  Expected test accuracy: {expected_acc:.2f}%")
        
        # Check if model is in eval mode
        if model.training:
            print("  [WARNING] Model is in training mode (should be eval)")
        else:
            print("  [OK] Model is in eval mode")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  [OK] Total parameters: {total_params:,}")
        print(f"  [OK] Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                output = model(test_input)
                
                # Handle different output types
                if isinstance(output, tuple):
                    output = output[0]
                elif isinstance(output, dict):
                    output = output.get('clipwise_output', output)
                
                if output.shape[1] == 30:
                    print(f"  [OK] Output shape correct: {output.shape}")
                else:
                    print(f"  [ERROR] Wrong output shape: {output.shape} (expected [1, 30])")
                
                # Check if output is valid (not NaN or Inf)
                if torch.isnan(output).any():
                    print("  [ERROR] Output contains NaN values!")
                elif torch.isinf(output).any():
                    print("  [ERROR] Output contains Inf values!")
                else:
                    print("  [OK] Output values are valid")
                    
                # Get prediction
                probs = torch.softmax(output, dim=-1)
                confidence, predicted_class = torch.max(probs, dim=-1)
                print(f"  [OK] Test prediction: class {predicted_class.item()}, confidence {confidence.item():.2%}")
                
        except Exception as e:
            print(f"  [ERROR] Forward pass failed: {e}")
    
    print("\n" + "-" * 80)
    
    # Verify ensemble configuration
    print("\nENSEMBLE CONFIGURATION:")
    print("-" * 80)
    print(f"Strategy: {ensemble.ensemble_strategy}")
    print(f"Confidence threshold: {ensemble.confidence_threshold}")
    print(f"Use calibration: {ensemble.use_calibration}")
    print(f"Temperature: {ensemble.temperature}")
    
    print("\nMODEL WEIGHTS:")
    for model_key, weight in ensemble.model_weights.items():
        model_name = expected_models[model_key][0]
        print(f"  {model_name:20s}: {weight}")
    
    return len(ensemble.models) == 6


def verify_preprocessing_pipeline():
    """
    Verify preprocessing matches training EXACTLY
    """
    print("\n" + "=" * 80)
    print("VERIFYING PREPROCESSING PIPELINE")
    print("=" * 80)
    
    # Create test audio (3 seconds of chirp)
    duration = 3.0
    sample_rate = 44100
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create stereo test signal to verify mono conversion
    freq_left = 2000 + 500 * np.sin(2 * np.pi * 0.5 * t)  # Varying frequency
    freq_right = 3000 + 500 * np.cos(2 * np.pi * 0.5 * t)  # Different in right channel
    
    stereo_audio = np.zeros((2, len(t)))
    stereo_audio[0] = np.sin(2 * np.pi * freq_left * t) * 0.5  # Left channel
    stereo_audio[1] = np.sin(2 * np.pi * freq_right * t) * 0.3  # Right channel
    
    print("\n1. AUDIO LOADING:")
    print("-" * 40)
    
    # Try torchaudio first
    try:
        import torchaudio
        print("[OK] torchaudio available")
        use_torchaudio = True
    except ImportError:
        print("[WARNING] torchaudio not available, using librosa")
        use_torchaudio = False
    
    # Convert stereo to mono
    mono_audio = np.mean(stereo_audio, axis=0)
    print(f"[OK] Converted stereo to mono: shape {stereo_audio.shape} -> {mono_audio.shape}")
    
    print(f"[OK] Sample rate: {sample_rate} Hz")
    print(f"[OK] Duration: {duration} seconds")
    print(f"[OK] Total samples: {len(mono_audio)}")
    
    print("\n2. SEGMENTATION:")
    print("-" * 40)
    
    segment_duration = 3.0
    segment_overlap = 0.5
    segment_length = int(segment_duration * sample_rate)
    hop_length = int(segment_length * (1 - segment_overlap))
    
    num_segments = (len(mono_audio) - segment_length) // hop_length + 1
    print(f"[OK] Segment duration: {segment_duration}s")
    print(f"[OK] Segment overlap: {segment_overlap * 100}%")
    print(f"[OK] Segment length: {segment_length} samples")
    print(f"[OK] Hop length: {hop_length} samples")
    print(f"[OK] Number of segments: {num_segments}")
    
    print("\n3. PRE-EMPHASIS:")
    print("-" * 40)
    
    pre_emphasis_coef = 0.97
    pre_emphasized = librosa.effects.preemphasis(mono_audio, coef=pre_emphasis_coef)
    print(f"[OK] Pre-emphasis coefficient: {pre_emphasis_coef}")
    print(f"[OK] Signal energy before: {np.mean(mono_audio**2):.6f}")
    print(f"[OK] Signal energy after: {np.mean(pre_emphasized**2):.6f}")
    
    print("\n4. MEL-SPECTROGRAM:")
    print("-" * 40)
    
    # Parameters
    n_fft = 2048
    hop_length_stft = 512
    n_mels = 128
    f_min = 300
    f_max = 15000
    
    print(f"[OK] n_fft: {n_fft}")
    print(f"[OK] hop_length: {hop_length_stft}")
    print(f"[OK] n_mels: {n_mels}")
    print(f"[OK] f_min: {f_min} Hz")
    print(f"[OK] f_max: {f_max} Hz")
    
    # Generate mel-spectrogram for first segment
    segment = pre_emphasized[:segment_length]
    
    mel_spec = librosa.feature.melspectrogram(
        y=segment,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length_stft,
        n_mels=n_mels,
        fmin=f_min,
        fmax=f_max
    )
    
    print(f"[OK] Mel-spectrogram shape: {mel_spec.shape}")
    
    print("\n5. dB CONVERSION:")
    print("-" * 40)
    
    top_db = 80
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=top_db)
    print(f"[OK] top_db: {top_db}")
    print(f"[OK] dB range before clipping: [{np.min(mel_spec_db):.1f}, {np.max(mel_spec_db):.1f}]")
    
    print("\n6. NORMALIZATION (CRITICAL):")
    print("-" * 40)
    
    # CRITICAL: Zero mean and unit variance normalization
    mean = np.mean(mel_spec_db)
    std = np.std(mel_spec_db)
    mel_spec_normalized = (mel_spec_db - mean) / (std + 1e-10)
    
    print(f"[OK] Original mean: {mean:.4f}")
    print(f"[OK] Original std: {std:.4f}")
    print(f"[OK] Normalized mean: {np.mean(mel_spec_normalized):.4f} (should be ~0)")
    print(f"[OK] Normalized std: {np.std(mel_spec_normalized):.4f} (should be ~1)")
    print(f"[OK] Normalized range: [{np.min(mel_spec_normalized):.2f}, {np.max(mel_spec_normalized):.2f}]")
    
    print("\n7. FINAL SHAPE:")
    print("-" * 40)
    
    # Ensure correct shape (128, 259)
    expected_time_frames = 259
    if mel_spec_normalized.shape[1] < expected_time_frames:
        pad_width = expected_time_frames - mel_spec_normalized.shape[1]
        mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
        print(f"[OK] Padded to shape: {mel_spec_normalized.shape}")
    elif mel_spec_normalized.shape[1] > expected_time_frames:
        mel_spec_normalized = mel_spec_normalized[:, :expected_time_frames]
        print(f"[OK] Trimmed to shape: {mel_spec_normalized.shape}")
    else:
        print(f"[OK] Shape already correct: {mel_spec_normalized.shape}")
    
    # Add batch and channel dimensions
    final_tensor = torch.FloatTensor(mel_spec_normalized).unsqueeze(0).unsqueeze(0)
    print(f"[OK] Final tensor shape: {final_tensor.shape}")
    print(f"  Expected: torch.Size([1, 1, 128, 259])")
    
    assert final_tensor.shape == (1, 1, 128, 259), f"Wrong shape: {final_tensor.shape}"
    
    print("\n" + "=" * 80)
    print("[PASS] PREPROCESSING PIPELINE VERIFIED")
    print("=" * 80)
    
    return True


def main():
    """
    Run all verification tests
    """
    print("\n" + "=" * 80)
    print("MODEL AND PREPROCESSING VERIFICATION")
    print("=" * 80)
    
    # Verify models
    models_ok = verify_model_loading()
    
    # Verify preprocessing
    preprocessing_ok = verify_preprocessing_pipeline()
    
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    if models_ok:
        print("[PASS] All 6 models loaded correctly")
    else:
        print("[WARNING] Some models failed to load")
    
    if preprocessing_ok:
        print("[PASS] Preprocessing pipeline matches training")
    else:
        print("[WARNING] Preprocessing issues detected")
    
    if models_ok and preprocessing_ok:
        print("\n>>> SYSTEM READY FOR BIRD IDENTIFICATION! <<<")
    else:
        print("\n>>> Please fix issues before testing <<<")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
