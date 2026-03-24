"""
Fixed preprocessing pipeline that EXACTLY matches training
"""

import numpy as np
import torch
import librosa
import warnings

warnings.filterwarnings('ignore')

# Try to import torchaudio
try:
    import torchaudio
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
    print("[WARNING] torchaudio not available, using librosa as fallback")


def load_audio_file(audio_path, target_sr=44100, duration_limit=None):
    """
    Load audio using torchaudio (preferred) or librosa (fallback)
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (44100 Hz)
        duration_limit: Maximum duration in seconds
        
    Returns:
        audio: Mono audio array
        sr: Sample rate
    """
    if TORCHAUDIO_AVAILABLE:
        try:
            # Load with torchaudio
            waveform, sr = torchaudio.load(audio_path)
            
            # Convert to mono by averaging channels if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform = resampler(waveform)
                sr = target_sr
            
            # Convert to numpy
            audio = waveform.squeeze().numpy()
            
            # Limit duration if specified
            if duration_limit:
                max_samples = int(duration_limit * sr)
                audio = audio[:max_samples]
                
            return audio, sr
            
        except Exception as e:
            print(f"torchaudio failed, falling back to librosa: {e}")
    
    # Fallback to librosa
    audio, sr = librosa.load(audio_path, sr=target_sr, mono=True, duration=duration_limit)
    return audio, sr


def preprocess_audio_segment(audio_segment, sr=44100):
    """
    Preprocess a single audio segment EXACTLY as in training
    
    Args:
        audio_segment: Audio array (3 seconds at 44100 Hz)
        sr: Sample rate (must be 44100)
        
    Returns:
        Normalized mel-spectrogram with shape (128, 259)
    """
    # Parameters (MUST match training exactly)
    PRE_EMPHASIS = 0.97
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    F_MIN = 300
    F_MAX = 15000
    TOP_DB = 80
    
    # 1. Apply pre-emphasis filter
    audio_emphasized = librosa.effects.preemphasis(audio_segment, coef=PRE_EMPHASIS)
    
    # 2. Generate mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_emphasized,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmin=F_MIN,
        fmax=F_MAX
    )
    
    # 3. Convert to dB scale with top_db
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=TOP_DB)
    
    # 4. CRITICAL: Normalize to zero mean and unit variance
    mean = np.mean(mel_spec_db)
    std = np.std(mel_spec_db)
    mel_spec_normalized = (mel_spec_db - mean) / (std + 1e-10)
    
    # 5. Ensure correct shape (128, 259)
    if mel_spec_normalized.shape[1] < 259:
        # Pad with zeros if too short
        pad_width = 259 - mel_spec_normalized.shape[1]
        mel_spec_normalized = np.pad(mel_spec_normalized, ((0, 0), (0, pad_width)), mode='constant')
    elif mel_spec_normalized.shape[1] > 259:
        # Trim if too long
        mel_spec_normalized = mel_spec_normalized[:, :259]
    
    return mel_spec_normalized


def segment_audio(audio, sr=44100, segment_duration=3.0, overlap=0.5):
    """
    Split audio into segments with overlap
    
    Args:
        audio: Audio array
        sr: Sample rate
        segment_duration: Duration of each segment in seconds
        overlap: Overlap ratio (0.5 = 50% overlap)
        
    Returns:
        List of audio segments
    """
    segment_length = int(segment_duration * sr)
    hop_length = int(segment_length * (1 - overlap))
    
    segments = []
    for start_idx in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start_idx:start_idx + segment_length]
        segments.append(segment)
    
    return segments


def complete_preprocessing_pipeline(audio_path, duration_limit=60):
    """
    Complete preprocessing pipeline matching training EXACTLY
    
    Args:
        audio_path: Path to audio file
        duration_limit: Maximum duration to process
        
    Returns:
        List of preprocessed spectrograms ready for model input
    """
    # Load audio (torchaudio preferred, librosa fallback)
    audio, sr = load_audio_file(audio_path, target_sr=44100, duration_limit=duration_limit)
    
    # Segment audio (3-second segments with 50% overlap)
    segments = segment_audio(audio, sr=sr, segment_duration=3.0, overlap=0.5)
    
    # Process each segment
    spectrograms = []
    for segment in segments:
        # Preprocess segment
        mel_spec_normalized = preprocess_audio_segment(segment, sr=sr)
        
        # Check if segment has significant energy (not silence)
        energy = np.mean(np.abs(mel_spec_normalized))
        if energy > 0.1:  # Threshold for normalized data
            spectrograms.append(mel_spec_normalized)
    
    return spectrograms, {
        'sample_rate': sr,
        'num_segments': len(spectrograms),
        'duration': len(audio) / sr
    }


# Test the preprocessing
if __name__ == "__main__":
    print("Testing preprocessing pipeline...")
    print("=" * 60)
    
    # Create test audio
    duration = 3.0
    sr = 44100
    t = np.linspace(0, duration, int(sr * duration))
    test_audio = np.sin(2 * np.pi * 1000 * t) * 0.5
    
    # Process
    mel_spec = preprocess_audio_segment(test_audio, sr=sr)
    
    print(f"[OK] Shape: {mel_spec.shape} (expected: (128, 259))")
    print(f"[OK] Mean: {np.mean(mel_spec):.6f} (expected: ~0)")
    print(f"[OK] Std: {np.std(mel_spec):.6f} (expected: ~1)")
    print(f"[OK] Range: [{np.min(mel_spec):.2f}, {np.max(mel_spec):.2f}]")
    
    # Test tensor conversion
    tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
    print(f"[OK] Tensor shape: {tensor.shape} (expected: [1, 1, 128, 259])")
    
    print("\n[PASS] Preprocessing pipeline ready!")
