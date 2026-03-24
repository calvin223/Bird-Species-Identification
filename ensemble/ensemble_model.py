"""
Ensemble Model for Bird Call Classification
Combines predictions from 6 different models using various strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import yaml
from collections import defaultdict
import warnings

from models.model_loader_direct import load_model_fixed
from models.model_architectures import get_model_info
from utils.audio_preprocessing import AudioPreprocessor

warnings.filterwarnings('ignore')


class BirdCallEnsemble:
    """
    Ensemble model combining 6 different architectures for bird call classification.
    Implements multiple ensemble strategies including weighted voting and confidence-based fusion.
    """
    
    def __init__(self, config_path: str = 'configs/ensemble_config.yaml'):
        """
        Initialize ensemble model with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device(self.config['inference']['device'] 
                                  if torch.cuda.is_available() 
                                  else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize audio preprocessor
        self.preprocessor = AudioPreprocessor(self.config)
        
        # Load species information
        self.species_names = self.config['species']['names']
        self.scientific_names = self.config['species'].get('scientific_names', {})
        self.num_classes = self.config['species']['num_classes']
        
        # Load models
        self.models = {}
        self.model_weights = {}
        self._load_all_models()
        
        # Ensemble parameters
        self.ensemble_strategy = self.config['ensemble']['strategy']
        self.confidence_threshold = self.config['ensemble']['confidence_threshold']
        self.use_calibration = self.config['ensemble']['use_calibration']
        self.temperature = self.config['ensemble']['temperature']
        
        # Statistics tracking
        self.model_stats = defaultdict(lambda: {
            'total_predictions': 0,
            'high_confidence_predictions': 0,
            'avg_confidence': 0.0
        })
    
    def _load_all_models(self):
        """Load all enabled models from configuration."""
        print("\nLoading ensemble models...")
        
        for model_key, model_config in self.config['models'].items():
            if not model_config['enabled']:
                print(f"Skipping {model_config['name']} (disabled)")
                continue
            
            try:
                print(f"Loading {model_config['name']}...")
                model = load_model_fixed(
                    model_type=model_config['type'],
                    model_path=model_config['path'],
                    device=self.device
                )
                self.models[model_key] = model
                self.model_weights[model_key] = model_config['weight']
                print(f"✓ {model_config['name']} loaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to load {model_config['name']}: {e}")
                if model_config.get('required', False):
                    raise
        
        print(f"\nSuccessfully loaded {len(self.models)}/{len(self.config['models'])} models")
    
    def predict_single_segment(self, spectrogram: torch.Tensor) -> Dict:
        """
        Get predictions from all models for a single spectrogram segment.
        
        Args:
            spectrogram: Input spectrogram tensor (1, 1, 128, 259)
            
        Returns:
            Dictionary with predictions from each model
        """
        predictions = {}
        
        with torch.no_grad():
            for model_key, model in self.models.items():
                try:
                    # Forward pass
                    output = model(spectrogram)
                    
                    # Handle different output formats
                    if model_key == 'model_4':  # AST returns tuple (logits, confidence)
                        if isinstance(output, tuple):
                            logits = output[0]
                        else:
                            logits = output
                    elif model_key == 'model_6':  # PANNs might return dict
                        if isinstance(output, dict):
                            logits = output['clipwise_output']
                        else:
                            logits = output
                    else:
                        logits = output
                    
                    # Apply temperature scaling if enabled
                    if self.use_calibration:
                        logits = logits / self.temperature
                    
                    # Get probabilities
                    probs = F.softmax(logits, dim=-1)
                    
                    # Get top predictions
                    confidence, predicted_class = torch.max(probs, dim=-1)
                    
                    # Get top-3 predictions
                    top3_probs, top3_classes = torch.topk(probs, k=min(3, self.num_classes), dim=-1)
                    
                    predictions[model_key] = {
                        'logits': logits.cpu().numpy(),
                        'probabilities': probs.cpu().numpy(),
                        'predicted_class': predicted_class.cpu().item(),
                        'confidence': confidence.cpu().item(),
                        'top3_classes': top3_classes.cpu().numpy(),
                        'top3_probs': top3_probs.cpu().numpy()
                    }
                    
                    # Update statistics
                    self.model_stats[model_key]['total_predictions'] += 1
                    self.model_stats[model_key]['avg_confidence'] += confidence.cpu().item()
                    if confidence.cpu().item() > self.confidence_threshold:
                        self.model_stats[model_key]['high_confidence_predictions'] += 1
                    
                except Exception as e:
                    print(f"Error in {model_key}: {e}")
                    predictions[model_key] = None
        
        return predictions
    
    def ensemble_predictions(self, all_predictions: List[Dict]) -> Dict:
        """
        Combine predictions from multiple segments and models.
        
        Args:
            all_predictions: List of prediction dictionaries for each segment
            
        Returns:
            Final ensemble prediction
        """
        if self.ensemble_strategy == 'weighted_average':
            return self._weighted_average_ensemble(all_predictions)
        elif self.ensemble_strategy == 'majority_voting':
            return self._majority_voting_ensemble(all_predictions)
        elif self.ensemble_strategy == 'stacking':
            return self._stacking_ensemble(all_predictions)
        else:
            return self._weighted_average_ensemble(all_predictions)
    
    def _weighted_average_ensemble(self, all_predictions: List[Dict]) -> Dict:
        """
        Weighted average ensemble strategy.
        
        Args:
            all_predictions: List of predictions from all segments
            
        Returns:
            Ensemble prediction
        """
        # Aggregate probabilities across all segments and models
        weighted_probs = np.zeros(self.num_classes)
        total_weight = 0
        
        model_predictions = defaultdict(list)
        
        for segment_pred in all_predictions:
            for model_key, pred in segment_pred.items():
                if pred is not None:
                    weight = self.model_weights[model_key]
                    probs = pred['probabilities'].squeeze()
                    weighted_probs += weight * probs
                    total_weight += weight
                    
                    # Store individual model predictions
                    model_predictions[model_key].append({
                        'class': pred['predicted_class'],
                        'confidence': pred['confidence']
                    })
        
        # Normalize
        if total_weight > 0:
            weighted_probs /= total_weight
        
        # Get final prediction
        predicted_class = np.argmax(weighted_probs)
        confidence = weighted_probs[predicted_class]
        
        # Calculate ensemble agreement
        agreement = self._calculate_agreement(model_predictions)
        
        # Get top-3 predictions
        top3_indices = np.argsort(weighted_probs)[-3:][::-1]
        top3_probs = weighted_probs[top3_indices]
        
        # Get scientific name if available
        species_name = self.species_names[predicted_class]
        scientific_name = self.scientific_names.get(species_name, "N/A")
        
        # Build top 3 predictions
        top_3_predictions = []
        for idx, prob in zip(top3_indices, top3_probs):
            sp_name = self.species_names[idx]
            top_3_predictions.append({
                'species': sp_name,
                'scientific_name': self.scientific_names.get(sp_name, "N/A"),
                'probability': float(prob)
            })
        
        return {
            'predicted_class': int(predicted_class),
            'species': species_name,
            'scientific_name': scientific_name,
            'confidence': float(confidence),
            'ensemble_agreement': float(agreement),
            'top_3_predictions': top_3_predictions,
            'all_probabilities': weighted_probs.tolist(),
            'model_predictions': self._summarize_model_predictions(model_predictions)
        }
    
    def _majority_voting_ensemble(self, all_predictions: List[Dict]) -> Dict:
        """
        Majority voting ensemble strategy.
        
        Args:
            all_predictions: List of predictions
            
        Returns:
            Ensemble prediction
        """
        # Count votes for each class
        vote_counts = defaultdict(float)
        model_predictions = defaultdict(list)
        
        for segment_pred in all_predictions:
            for model_key, pred in segment_pred.items():
                if pred is not None:
                    predicted_class = pred['predicted_class']
                    weight = self.model_weights[model_key]
                    vote_counts[predicted_class] += weight
                    
                    model_predictions[model_key].append({
                        'class': predicted_class,
                        'confidence': pred['confidence']
                    })
        
        # Find class with most votes
        if vote_counts:
            predicted_class = max(vote_counts, key=vote_counts.get)
            total_votes = sum(vote_counts.values())
            confidence = vote_counts[predicted_class] / total_votes if total_votes > 0 else 0
        else:
            predicted_class = 0
            confidence = 0.0
        
        # Calculate agreement
        agreement = self._calculate_agreement(model_predictions)
        
        # Get scientific name if available
        species_name = self.species_names[predicted_class]
        scientific_name = self.scientific_names.get(species_name, "N/A")
        
        return {
            'predicted_class': int(predicted_class),
            'species': species_name,
            'scientific_name': scientific_name,
            'confidence': float(confidence),
            'ensemble_agreement': float(agreement),
            'vote_distribution': dict(vote_counts),
            'model_predictions': self._summarize_model_predictions(model_predictions)
        }
    
    def _stacking_ensemble(self, all_predictions: List[Dict]) -> Dict:
        """
        Stacking ensemble strategy (simplified version without meta-learner).
        Uses confidence-weighted combination.
        
        Args:
            all_predictions: List of predictions
            
        Returns:
            Ensemble prediction
        """
        # Use confidence-weighted averaging as a simple stacking approach
        weighted_probs = np.zeros(self.num_classes)
        total_confidence = 0
        
        for segment_pred in all_predictions:
            for model_key, pred in segment_pred.items():
                if pred is not None:
                    confidence = pred['confidence']
                    probs = pred['probabilities'].squeeze()
                    
                    # Weight by both model weight and confidence
                    weight = self.model_weights[model_key] * confidence
                    weighted_probs += weight * probs
                    total_confidence += weight
        
        # Normalize
        if total_confidence > 0:
            weighted_probs /= total_confidence
        
        predicted_class = np.argmax(weighted_probs)
        confidence = weighted_probs[predicted_class]
        
        # Get scientific name if available
        species_name = self.species_names[predicted_class]
        scientific_name = self.scientific_names.get(species_name, "N/A")
        
        return {
            'predicted_class': int(predicted_class),
            'species': species_name,
            'scientific_name': scientific_name,
            'confidence': float(confidence),
            'all_probabilities': weighted_probs.tolist()
        }
    
    def _calculate_agreement(self, model_predictions: Dict) -> float:
        """
        Calculate agreement score among models.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Agreement score (0-1)
        """
        if not model_predictions:
            return 0.0
        
        # Get most common prediction for each model
        model_votes = {}
        for model_key, preds in model_predictions.items():
            if preds:
                # Get most common class prediction for this model
                classes = [p['class'] for p in preds]
                most_common = max(set(classes), key=classes.count)
                model_votes[model_key] = most_common
        
        if not model_votes:
            return 0.0
        
        # Calculate agreement
        all_votes = list(model_votes.values())
        if all_votes:
            most_common_vote = max(set(all_votes), key=all_votes.count)
            agreement = all_votes.count(most_common_vote) / len(all_votes)
            return agreement
        
        return 0.0
    
    def _summarize_model_predictions(self, model_predictions: Dict) -> Dict:
        """
        Summarize predictions from each model.
        
        Args:
            model_predictions: Dictionary of model predictions
            
        Returns:
            Summary dictionary
        """
        summary = {}
        
        for model_key, preds in model_predictions.items():
            if preds:
                # Get most common prediction
                classes = [p['class'] for p in preds]
                confidences = [p['confidence'] for p in preds]
                
                if classes:
                    most_common_class = max(set(classes), key=classes.count)
                    avg_confidence = np.mean(confidences)
                    
                    model_name = self.config['models'][model_key]['name']
                    summary[model_name] = {
                        'predicted_species': self.species_names[most_common_class],
                        'average_confidence': float(avg_confidence),
                        'num_segments': len(preds)
                    }
        
        return summary
    
    def predict_audio_file(self, audio_path: Union[str, Path]) -> Dict:
        """
        Complete prediction pipeline for an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Ensemble prediction with detailed results
        """
        print(f"\nProcessing: {audio_path}")
        
        # Preprocess audio
        spectrograms, metadata = self.preprocessor.process_audio_file(audio_path)
        print(f"Generated {len(spectrograms)} segments from {metadata['duration_seconds']:.2f}s audio")
        
        # Batch spectrograms
        batches = self.preprocessor.batch_spectrograms(
            spectrograms, 
            batch_size=self.config['inference']['batch_size']
        )
        
        # Get predictions for all segments
        all_predictions = []
        
        for batch_idx, batch in enumerate(batches):
            batch = batch.to(self.device)
            
            # Process each segment in batch
            for i in range(batch.shape[0]):
                segment = batch[i:i+1]
                segment_predictions = self.predict_single_segment(segment)
                all_predictions.append(segment_predictions)
        
        # Combine predictions using ensemble strategy
        ensemble_result = self.ensemble_predictions(all_predictions)
        
        # Add metadata
        ensemble_result['metadata'] = metadata
        ensemble_result['num_models'] = len(self.models)
        ensemble_result['ensemble_strategy'] = self.ensemble_strategy
        
        # Print result
        print(f"\nPrediction: {ensemble_result['species']}")
        print(f"Scientific name: {ensemble_result['scientific_name']}")
        print(f"Confidence: {ensemble_result['confidence']:.2%}")
        print(f"Ensemble agreement: {ensemble_result['ensemble_agreement']:.2%}")
        
        return ensemble_result
    
    def save_model_stats(self, output_path: str):
        """
        Save model statistics to file.
        
        Args:
            output_path: Path to save statistics
        """
        stats = {}
        for model_key, model_stat in self.model_stats.items():
            total = model_stat['total_predictions']
            if total > 0:
                stats[model_key] = {
                    'model_name': self.config['models'][model_key]['name'],
                    'total_predictions': total,
                    'high_confidence_ratio': model_stat['high_confidence_predictions'] / total,
                    'average_confidence': model_stat['avg_confidence'] / total
                }
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Model statistics saved to {output_path}")


class EnsemblePredictor:
    """
    High-level interface for ensemble predictions.
    """
    
    def __init__(self, config_path: str = 'configs/ensemble_config.yaml'):
        """
        Initialize predictor with ensemble model.
        
        Args:
            config_path: Path to configuration file
        """
        self.ensemble = BirdCallEnsemble(config_path)
        self.results_history = []
    
    def predict(self, audio_path: Union[str, Path]) -> Dict:
        """
        Predict bird species from audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Prediction results
        """
        result = self.ensemble.predict_audio_file(audio_path)
        self.results_history.append(result)
        return result
    
    def predict_batch(self, audio_paths: List[Union[str, Path]]) -> List[Dict]:
        """
        Predict bird species for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths
            
        Returns:
            List of prediction results
        """
        results = []
        for audio_path in audio_paths:
            try:
                result = self.predict(audio_path)
                results.append(result)
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                results.append({
                    'error': str(e),
                    'audio_path': str(audio_path)
                })
        return results
    
    def save_results(self, output_path: str):
        """
        Save all prediction results to file.
        
        Args:
            output_path: Path to save results
        """
        with open(output_path, 'w') as f:
            json.dump(self.results_history, f, indent=2)
        print(f"Results saved to {output_path}")
