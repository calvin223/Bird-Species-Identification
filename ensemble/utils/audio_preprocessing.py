"""
Audio Preprocessing Module for Bird Call Classification Ensemble
Handles audio loading, segmentation, and mel-spectrogram generation
"""

import numpy as np
import torch
import torchaudio
import librosa
import warnings
from typing import Tuple, List, Optional, Union
from pathlib import Path

warnings.filterwarnings('ignore')


class AudioPreprocessor:
    """
    Preprocesses audio files for bird call classification.
    Ensures exact match with training preprocessing pipeline.
    """
    
    def __init__(self, config: dict):
        """
        Initialize audio preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
        """
        self.sample_rate = config['audio']['sample_rate']
        self.segment_duration = config['audio']['segment_duration']
        self.segment_overlap = config['audio']['segment_overlap']
        self.pre_emphasis_coeff = config['audio']['pre_emphasis_coeff']
        
        # Mel-spectrogram parameters
        self.n_fft = config['spectrogram']['n_fft']
        self.hop_length = config['spectrogram']['hop_length']
        self.n_mels = config['spectrogram']['n_mels']
        self.f_min = config['spectrogram']['f_min']
        self.f_max = config['spectrogram']['f_max']
        self.top_db = config['spectrogram']['top_db']
        
        # Calculate segment parameters
        self.segment_samples = int(self.segment_duration * self.sample_rate)
        self.overlap_samples = int(self.segment_samples * self.segment_overlap)
        self.hop_samples = self.segment_samples - self.overlap_samples
        
    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """
        Load audio file using torchaudio with librosa fallback.
        
        Args:
            audio_path: Path to audio file (MP3 or WAV)
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_path = Path(audio_path)
        
        try:
            # Try loading with torchaudio first
            waveform, sr = torchaudio.load(str(audio_path))
            audio = waveform.numpy()
        except Exception as e:
            print(f"torchaudio failed, falling back to librosa: {e}")
            # Fallback to librosa
            audio, sr = librosa.load(str(audio_path), sr=None, mono=False)
            if audio.ndim == 1:
                audio = audio[np.newaxis, :]
        
        return audio, sr
    
    def resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """
        Resample audio to target sample rate if needed.
        
        Args:
            audio: Audio array
            orig_sr: Original sample rate
            
        Returns:
            Resampled audio array
        """
        if orig_sr != self.sample_rate:
            # Convert to torch tensor for resampling
            audio_tensor = torch.from_numpy(audio).float()
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.numpy()
        
        return audio
    
    def convert_to_mono(self, audio: np.ndarray) -> np.ndarray:
        """
        Convert stereo audio to mono by averaging channels.
        
        Args:
            audio: Audio array (channels, samples)
            
        Returns:
            Mono audio array (1, samples)
        """
        if audio.shape[0] > 1:
            audio = np.mean(audio, axis=0, keepdims=True)
        return audio
    
    def apply_pre_emphasis(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply pre-emphasis filter to enhance high frequencies.
        
        Args:
            audio: Audio array
            
        Returns:
            Pre-emphasized audio
        """
        emphasized = np.copy(audio)
        emphasized[:, 1:] = audio[:, 1:] - self.pre_emphasis_coeff * audio[:, :-1]
        emphasized[:, 0] = audio[:, 0]
        return emphasized
    
    def segment_audio(self, audio: np.ndarray) -> List[np.ndarray]:
        """
        Split audio into overlapping segments.
        
        Args:
            audio: Audio array (1, samples)
            
        Returns:
            List of audio segments
        """
        audio = audio.flatten()
        segments = []
        
        # Calculate number of segments
        total_samples = len(audio)
        if total_samples < self.segment_samples:
            # Pad if audio is shorter than segment duration
            padding = self.segment_samples - total_samples
            audio = np.pad(audio, (0, padding), mode='constant')
            segments.append(audio)
        else:
            # Create overlapping segments
            start = 0
            while start + self.segment_samples <= total_samples:
                segment = audio[start:start + self.segment_samples]
                segments.append(segment)
                start += self.hop_samples
            
            # Handle last segment if there's remaining audio
            if start < total_samples and (total_samples - start) > self.sample_rate:
                # Only include if remaining is more than 1 second
                last_segment = audio[-self.segment_samples:]
                segments.append(last_segment)
        
        return segments
    
    def compute_mel_spectrogram(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Compute mel-spectrogram from audio segment.
        
        Args:
            audio_segment: Audio segment array
            
        Returns:
            Normalized mel-spectrogram (1, n_mels, time_frames)
        """
        # Compute mel-spectrogram using librosa
        mel_spec = librosa.feature.melspectrogram(
            y=audio_segment,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
        
        # Normalize to zero mean and unit variance
        mean = np.mean(mel_spec_db)
        std = np.std(mel_spec_db)
        if std > 0:
            mel_spec_db = (mel_spec_db - mean) / std
        else:
            mel_spec_db = mel_spec_db - mean
        
        # Add channel dimension
        mel_spec_db = mel_spec_db[np.newaxis, :, :]
        
        return mel_spec_db
    
    def process_audio_file(self, audio_path: Union[str, Path]) -> Tuple[List[np.ndarray], dict]:
        """
        Complete preprocessing pipeline for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (list of mel-spectrograms, metadata dict)
        """
        # Load audio
        audio, orig_sr = self.load_audio(audio_path)
        
        # Resample if needed
        audio = self.resample_audio(audio, orig_sr)
        
        # Convert to mono
        audio = self.convert_to_mono(audio)
        
        # Apply pre-emphasis
        audio = self.apply_pre_emphasis(audio)
        
        # Segment audio
        segments = self.segment_audio(audio)
        
        # Compute mel-spectrograms for all segments
        spectrograms = []
        for segment in segments:
            mel_spec = self.compute_mel_spectrogram(segment)
            spectrograms.append(mel_spec)
        
        # Metadata
        metadata = {
            'original_sample_rate': orig_sr,
            'duration_seconds': len(audio.flatten()) / self.sample_rate,
            'num_segments': len(spectrograms),
            'segment_duration': self.segment_duration,
            'overlap': self.segment_overlap,
            'audio_path': str(audio_path)
        }
        
        return spectrograms, metadata
    
    def batch_spectrograms(self, spectrograms: List[np.ndarray], 
                          batch_size: int = 32) -> List[torch.Tensor]:
        """
        Convert list of numpy spectrograms to batched torch tensors.
        
        Args:
            spectrograms: List of mel-spectrograms
            batch_size: Batch size for processing
            
        Returns:
            List of batched torch tensors
        """
        batches = []
        
        for i in range(0, len(spectrograms), batch_size):
            batch = spectrograms[i:i + batch_size]
            batch_tensor = torch.FloatTensor(np.array(batch))
            batches.append(batch_tensor)
        
        return batches


class SpecAugment:
    """
    Spec augmentation for test-time augmentation (optional).
    Based on SpecAugment paper: https://arxiv.org/abs/1904.08779
    """
    
    def __init__(self, time_mask_max: int = 30, freq_mask_max: int = 20,
                 n_time_masks: int = 2, n_freq_masks: int = 2):
        """
        Initialize SpecAugment.
        
        Args:
            time_mask_max: Maximum time mask length
            freq_mask_max: Maximum frequency mask length
            n_time_masks: Number of time masks
            n_freq_masks: Number of frequency masks
        """
        self.time_mask_max = time_mask_max
        self.freq_mask_max = freq_mask_max
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
    
    def __call__(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.
        
        Args:
            spectrogram: Input spectrogram tensor
            
        Returns:
            Augmented spectrogram
        """
        spec = spectrogram.clone()
        _, n_mels, n_frames = spec.shape
        
        # Time masking
        for _ in range(self.n_time_masks):
            t = np.random.randint(0, min(self.time_mask_max, n_frames))
            t0 = np.random.randint(0, n_frames - t) if n_frames - t > 0 else 0
            spec[:, :, t0:t0 + t] = 0
        
        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = np.random.randint(0, min(self.freq_mask_max, n_mels))
            f0 = np.random.randint(0, n_mels - f) if n_mels - f > 0 else 0
            spec[:, f0:f0 + f, :] = 0
        
        return spec


def validate_spectrogram_shape(spectrogram: np.ndarray, expected_shape: Tuple[int, int, int]) -> bool:
    """
    Validate spectrogram shape matches expected dimensions.
    
    Args:
        spectrogram: Mel-spectrogram array
        expected_shape: Expected shape (channels, n_mels, time_frames)
        
    Returns:
        True if shape matches or is compatible
    """
    if spectrogram.shape == expected_shape:
        return True
    
    # Allow slight variations in time dimension
    if (spectrogram.shape[0] == expected_shape[0] and 
        spectrogram.shape[1] == expected_shape[1] and
        abs(spectrogram.shape[2] - expected_shape[2]) <= 5):
        return True
    
    return False
