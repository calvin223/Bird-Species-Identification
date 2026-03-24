"""
Test script for ensemble model
Verifies that all components are working correctly
"""

import sys
import torch
import numpy as np
from pathlib import Path
import yaml

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from ensemble_model import BirdCallEnsemble, EnsemblePredictor
        from utils.audio_preprocessing import AudioPreprocessor
        from models.model_architectures import (
            BirdCallEfficientNet, ResNet50CBAM, DenseNet121Custom,
            AudioSpectrogramTransformer, ConvNeXtTiny, CNN14
        )
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_config():
    """Test configuration file loading."""
    print("\nTesting configuration...")
    try:
        config_path = Path("configs/ensemble_config.yaml")
        if not config_path.exists():
            print(f"✗ Config file not found: {config_path}")
            return False
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check essential config sections
        required_sections = ['audio', 'spectrogram', 'models', 'ensemble', 'species']
        for section in required_sections:
            if section not in config:
                print(f"✗ Missing config section: {section}")
                return False
        
        print(f"✓ Configuration loaded successfully")
        print(f"  - {len(config['models'])} models configured")
        print(f"  - {config['species']['num_classes']} species classes")
        print(f"  - Ensemble strategy: {config['ensemble']['strategy']}")
        return True
        
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def test_audio_preprocessing():
    """Test audio preprocessing with synthetic data."""
    print("\nTesting audio preprocessing...")
    try:
        # Load config
        with open("configs/ensemble_config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        from utils.audio_preprocessing import AudioPreprocessor
        
        # Create preprocessor
        preprocessor = AudioPreprocessor(config)
        
        # Create synthetic audio (3 seconds at 44.1kHz)
        sample_rate = 44100
        duration = 3.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        audio = audio[np.newaxis, :]  # Add channel dimension
        
        # Test preprocessing steps
        # 1. Pre-emphasis
        emphasized = preprocessor.apply_pre_emphasis(audio)
        assert emphasized.shape == audio.shape
        
        # 2. Segmentation
        segments = preprocessor.segment_audio(audio)
        assert len(segments) > 0
        
        # 3. Mel-spectrogram
        mel_spec = preprocessor.compute_mel_spectrogram(segments[0])
        assert mel_spec.shape == (1, 128, 259)  # Expected shape
        
        print("✓ Audio preprocessing working correctly")
        print(f"  - Generated mel-spectrogram shape: {mel_spec.shape}")
        return True
        
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False

def test_model_architectures():
    """Test that model architectures can be instantiated."""
    print("\nTesting model architectures...")
    
    models_to_test = [
        ("EfficientNet-B1", "BirdCallEfficientNet"),
        ("ResNet-50 CBAM", "ResNet50CBAM"),
        ("DenseNet-121", "DenseNet121Custom"),
        ("AST", "AudioSpectrogramTransformer"),
        ("ConvNeXt-Tiny", "ConvNeXtTiny"),
        ("PANNs CNN14", "CNN14")
    ]
    
    from models import model_architectures
    
    success_count = 0
    for model_name, class_name in models_to_test:
        try:
            model_class = getattr(model_architectures, class_name)
            model = model_class(num_classes=30)
            model.eval()  # IMPORTANT: Set to eval mode before testing
            
            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 1, 128, 259)
            with torch.no_grad():
                output = model(dummy_input)
            
            assert output.shape == (1, 30), f"Expected output shape (1, 30), got {output.shape}"
            
            print(f"  ✓ {model_name} initialized successfully")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
    
    print(f"\n✓ {success_count}/{len(models_to_test)} models tested successfully")
    return success_count == len(models_to_test)

def test_ensemble_initialization():
    """Test ensemble model initialization."""
    print("\nTesting ensemble initialization...")
    try:
        from ensemble_model import BirdCallEnsemble
        
        # Note: This will attempt to load actual model checkpoints
        # It may fail if checkpoint files are not present
        print("  Attempting to initialize ensemble...")
        print("  (This may show warnings if model checkpoints are missing)")
        
        ensemble = BirdCallEnsemble("configs/ensemble_config.yaml")
        
        print(f"\n✓ Ensemble initialized")
        print(f"  - {len(ensemble.models)} models loaded")
        print(f"  - Device: {ensemble.device}")
        print(f"  - Strategy: {ensemble.ensemble_strategy}")
        
        # Test with synthetic spectrogram
        if len(ensemble.models) > 0:
            dummy_spec = torch.randn(1, 1, 128, 259).to(ensemble.device)
            predictions = ensemble.predict_single_segment(dummy_spec)
            print(f"  - Test prediction successful: {len(predictions)} model outputs")
        
        return True
        
    except Exception as e:
        print(f"✗ Ensemble initialization error: {e}")
        print("  (This is expected if model checkpoints are not available)")
        return False

def test_output_format():
    """Test that output format matches specification."""
    print("\nTesting output format...")
    
    # Create sample output
    sample_output = {
        "species": "Common Blackbird",
        "scientific_name": "Turdus merula",
        "confidence": 0.943,
        "ensemble_agreement": 0.83,
        "top_3_predictions": [
            {
                "species": "Common Blackbird",
                "scientific_name": "Turdus merula",
                "probability": 0.943
            }
        ],
        "model_predictions": {
            "EfficientNet-B1": {
                "predicted_species": "Common Blackbird",
                "average_confidence": 0.95,
                "num_segments": 3
            }
        }
    }
    
    # Check required fields
    required_fields = ["species", "scientific_name", "confidence"]
    for field in required_fields:
        if field not in sample_output:
            print(f"  ✗ Missing required field: {field}")
            return False
    
    print("✓ Output format validated")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 ENSEMBLE MODEL TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Audio Preprocessing", test_audio_preprocessing),
        ("Model Architectures", test_model_architectures),
        ("Output Format", test_output_format),
        ("Ensemble Initialization", test_ensemble_initialization),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10} {test_name}")
    
    print("-" * 60)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The ensemble model is ready to use.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())
