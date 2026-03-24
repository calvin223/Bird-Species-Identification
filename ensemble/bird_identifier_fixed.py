"""
Bird Call Identification System - FIXED VERSION
Corrects ensemble voting to properly aggregate model predictions
"""

import sys
import torch
import numpy as np
from pathlib import Path
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings('ignore')
sys.path.append(str(Path(__file__).parent))

from ensemble_model import BirdCallEnsemble
from preprocessing_fixed import load_audio_file, preprocess_audio_segment, segment_audio


def identify_bird(audio_path, show_details=True, use_majority_voting=True):
    """
    Identify bird species from audio recording with fixed voting logic
    
    Args:
        audio_path: Path to audio file
        show_details: Show detailed model performance
        use_majority_voting: If True, use simple majority voting across models
    
    Returns:
        tuple: (predicted_species, confidence)
    """
    print("\n" + "=" * 80)
    print("BIRD SPECIES IDENTIFICATION SYSTEM - FIXED")
    print("6-Model Ensemble | Corrected Voting Logic")
    print("=" * 80)
    
    # Load ensemble
    print("\nLoading models...")
    ensemble = BirdCallEnsemble('configs/ensemble_config.yaml')
    device = ensemble.device
    print(f"Device: {device}")
    print(f"Models loaded: {len(ensemble.models)}/6")
    
    # Load audio
    print(f"\nProcessing: {Path(audio_path).name}")
    try:
        audio, sr = load_audio_file(audio_path, target_sr=44100, duration_limit=30)
        duration = len(audio) / sr
        print(f"Duration: {duration:.1f}s | Sample rate: {sr}Hz")
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, 0
    
    # Segment audio
    audio_segments = segment_audio(audio, sr=sr, segment_duration=3.0, overlap=0.5)
    print(f"Segments: {len(audio_segments)} (3s each, 50% overlap)")
    
    # Initialize tracking
    all_model_predictions = defaultdict(list)  # model -> list of predictions
    all_model_confidences = defaultdict(list)  # model -> list of confidences
    segment_wise_predictions = []  # For segment-level ensemble
    
    print("\nAnalyzing segments...")
    
    # Process each segment
    valid_segments = 0
    for segment in audio_segments:
        mel_spec_normalized = preprocess_audio_segment(segment, sr=sr)
        
        # Skip silent segments
        if np.mean(np.abs(mel_spec_normalized)) < 0.1:
            continue
        
        valid_segments += 1
        
        # Convert to tensor
        spec_tensor = torch.FloatTensor(mel_spec_normalized).unsqueeze(0).unsqueeze(0).to(device)
        
        # Get predictions from each model
        segment_pred = ensemble.predict_single_segment(spec_tensor)
        
        # Store individual model predictions
        for model_key, pred in segment_pred.items():
            if pred is not None:
                all_model_predictions[model_key].append(pred['predicted_class'])
                all_model_confidences[model_key].append(pred['confidence'])
        
        # Also get ensemble prediction for this segment
        ensemble_result = ensemble.ensemble_predictions([segment_pred])
        segment_wise_predictions.append(ensemble_result['predicted_class'])
    
    if valid_segments == 0:
        print("No valid segments found")
        return None, 0
    
    print(f"Analyzed {valid_segments} valid segments")
    
    # FIXED: Aggregate predictions using majority voting across models
    model_final_predictions = {}
    model_final_confidences = {}
    
    for model_key in all_model_predictions:
        if all_model_predictions[model_key]:
            # Get most common prediction for this model across all segments
            model_votes = Counter(all_model_predictions[model_key])
            most_common = model_votes.most_common(1)[0]
            model_final_predictions[model_key] = most_common[0]
            
            # Calculate average confidence for this prediction
            indices = [i for i, pred in enumerate(all_model_predictions[model_key]) if pred == most_common[0]]
            model_final_confidences[model_key] = np.mean([all_model_confidences[model_key][i] for i in indices])
    
    # FIXED: Use majority voting across models
    if use_majority_voting:
        # Simple majority voting: each model gets one vote
        model_species_votes = Counter(model_final_predictions.values())
        most_common_species_idx = model_species_votes.most_common(1)[0][0]
        
        # Calculate how many models voted for the winning species
        models_voting_for_winner = model_species_votes[most_common_species_idx]
        
        # Average confidence from models that voted for the winner
        winner_confidences = [conf for model, conf in model_final_confidences.items() 
                             if model_final_predictions[model] == most_common_species_idx]
        avg_confidence = np.mean(winner_confidences) if winner_confidences else 0
        
    else:
        # Use weighted ensemble voting (original method)
        species_votes = Counter(segment_wise_predictions)
        most_common_species_idx = species_votes.most_common(1)[0][0]
        models_voting_for_winner = sum(1 for pred in model_final_predictions.values() 
                                      if pred == most_common_species_idx)
        
        # Calculate average confidence
        all_confidences = []
        for model_confs in all_model_confidences.values():
            all_confidences.extend(model_confs)
        avg_confidence = np.mean(all_confidences) if all_confidences else 0
    
    # Get species name
    predicted_species = ensemble.species_names[most_common_species_idx]
    scientific_name = ensemble.scientific_names.get(predicted_species, "N/A")
    
    # Display results
    print("\n" + "=" * 80)
    print("IDENTIFICATION RESULTS")
    print("=" * 80)
    
    print(f"\nVoting Method: {'Majority Voting' if use_majority_voting else 'Weighted Ensemble'}")
    
    if avg_confidence < 0.35 and models_voting_for_winner < 3:
        print(f"\nLOW CONFIDENCE - Likely out of scope")
        print(f"Species: {predicted_species}")
        print(f"Confidence: {avg_confidence:.1%}")
        print(f"Models agreeing: {models_voting_for_winner}/6")
    else:
        print(f"\nSpecies: {predicted_species}")
        print(f"Scientific name: {scientific_name}")
        print(f"Confidence: {avg_confidence:.1%}")
        print(f"Models agreeing: {models_voting_for_winner}/6")
    
    if show_details:
        # Model information
        model_info = {
            'model_1': ('EfficientNet-B1', 95.77),
            'model_2': ('ResNet-50 CBAM', 98.15),
            'model_3': ('DenseNet-121', 86.33),
            'model_4': ('AST', 85.60),
            'model_5': ('ConvNeXt-Tiny', 96.52),
            'model_6': ('PANNs CNN14', 88.59)
        }
        
        print("\n" + "-" * 80)
        print("MODEL VOTES")
        print("-" * 80)
        print(f"{'Model':<20} {'Test Acc':<10} {'Prediction':<20} {'Avg Confidence':<15}")
        print("-" * 80)
        
        # Count species votes
        species_vote_count = Counter()
        
        for model_key in ['model_1', 'model_2', 'model_3', 'model_4', 'model_5', 'model_6']:
            if model_key in model_final_predictions:
                model_name, test_acc = model_info[model_key]
                pred_idx = model_final_predictions[model_key]
                model_species = ensemble.species_names[pred_idx]
                model_conf = model_final_confidences[model_key]
                
                species_vote_count[model_species] += 1
                
                # Highlight if this model voted for the winner
                marker = " <--" if pred_idx == most_common_species_idx else ""
                
                print(f"{model_name:<20} {test_acc:<10.1f}% {model_species:<20} "
                      f"{model_conf:<15.1%}{marker}")
        
        # Show vote summary
        print("\n" + "-" * 80)
        print("SPECIES VOTE COUNT")
        print("-" * 80)
        for species, count in species_vote_count.most_common():
            print(f"{species:<30} {count} votes")
        
        # Consensus analysis
        print(f"\nConsensus Level: {models_voting_for_winner}/6 models")
        
        if models_voting_for_winner >= 5:
            print("Strong consensus - High reliability")
        elif models_voting_for_winner >= 4:
            print("Good consensus - Reliable")
        elif models_voting_for_winner >= 3:
            print("Moderate consensus")
        else:
            print("Weak consensus - Uncertain")
    
    print("\n" + "=" * 80)
    
    return predicted_species, avg_confidence


if __name__ == "__main__":
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
    else:
        print("Usage: python bird_identifier_fixed.py <audio_file>")
        sys.exit(1)
    
    if not Path(audio_file).exists():
        print(f"Error: File not found: {audio_file}")
        sys.exit(1)
    
    # Test with majority voting (fixed method)
    print("\n### USING MAJORITY VOTING (FIXED) ###")
    species, confidence = identify_bird(audio_file, show_details=True, use_majority_voting=True)
    
    if species:
        print(f"\nFinal (Majority): {species} ({confidence:.1%} confidence)")
