"""
Evaluate the ensemble model on the test split
Generates detailed performance metrics and confusion matrix
"""

import sys
import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime
import warnings
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from ensemble_model import BirdCallEnsemble


class EnsembleTestEvaluator:
    """Evaluator for testing ensemble on test dataset"""
    
    def __init__(self, ensemble_path='configs/ensemble_config.yaml'):
        """Initialize evaluator with ensemble model"""
        print("Initializing ensemble model...")
        self.ensemble = BirdCallEnsemble(ensemble_path)
        self.device = self.ensemble.device
        
        # Data paths
        self.test_features_path = r"D:\University\Comp702\02_data\bird_calls_30species_memmap\test_features.dat"
        self.test_labels_path = r"D:\University\Comp702\02_data\bird_calls_30species_memmap\test_labels.npy"
        
        # Load test data info
        self._load_test_data()
        
        # Results storage
        self.predictions = []
        self.true_labels = []
        self.confidences = []
        self.model_predictions = {key: [] for key in self.ensemble.models.keys()}
        
    def _load_test_data(self):
        """Load test dataset information"""
        print("\nLoading test dataset...")
        
        # Load labels
        self.test_labels = np.load(self.test_labels_path)
        self.num_samples = len(self.test_labels)
        
        # Load memory-mapped features
        self.test_features = np.memmap(
            self.test_features_path,
            dtype='float32',
            mode='r',
            shape=(self.num_samples, 1, 128, 259)
        )
        
        print(f"Test set size: {self.num_samples} samples")
        print(f"Number of classes: {self.ensemble.num_classes}")
        print(f"Feature shape: {self.test_features[0].shape}")
        
        # Calculate class distribution
        unique, counts = np.unique(self.test_labels, return_counts=True)
        self.class_distribution = dict(zip(unique, counts))
        
    def evaluate_batch(self, batch_indices):
        """Evaluate a batch of samples"""
        batch_predictions = []
        batch_confidences = []
        batch_model_preds = {key: [] for key in self.ensemble.models.keys()}
        
        for idx in batch_indices:
            # Get the spectrogram
            spectrogram = self.test_features[idx:idx+1]
            spectrogram_tensor = torch.FloatTensor(spectrogram).to(self.device)
            
            # Get predictions from all models
            segment_pred = self.ensemble.predict_single_segment(spectrogram_tensor)
            
            # Use ensemble strategy to combine
            ensemble_result = self.ensemble.ensemble_predictions([segment_pred])
            
            # Store results
            batch_predictions.append(ensemble_result['predicted_class'])
            batch_confidences.append(ensemble_result['confidence'])
            
            # Store individual model predictions
            for model_key, pred in segment_pred.items():
                if pred is not None:
                    batch_model_preds[model_key].append(pred['predicted_class'])
                else:
                    batch_model_preds[model_key].append(-1)  # Error marker
        
        return batch_predictions, batch_confidences, batch_model_preds
    
    def run_evaluation(self, batch_size=32, max_samples=None):
        """Run full evaluation on test set"""
        print("\n" + "="*80)
        print("STARTING ENSEMBLE EVALUATION ON TEST SET")
        print("="*80)
        
        # Determine number of samples to process
        num_samples = min(max_samples, self.num_samples) if max_samples else self.num_samples
        print(f"\nEvaluating {num_samples} test samples...")
        
        # Process in batches
        start_time = time.time()
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = range(start_idx, end_idx)
            
            # Process batch
            batch_preds, batch_confs, batch_model_preds = self.evaluate_batch(batch_indices)
            
            # Store results
            self.predictions.extend(batch_preds)
            self.confidences.extend(batch_confs)
            self.true_labels.extend(self.test_labels[start_idx:end_idx])
            
            for model_key in self.ensemble.models.keys():
                self.model_predictions[model_key].extend(batch_model_preds[model_key])
            
            # Progress update
            if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
                progress = (batch_idx + 1) / num_batches * 100
                elapsed = time.time() - start_time
                eta = (elapsed / (batch_idx + 1)) * (num_batches - batch_idx - 1)
                print(f"Progress: {progress:.1f}% | Batch {batch_idx+1}/{num_batches} | "
                      f"ETA: {eta:.0f}s")
        
        total_time = time.time() - start_time
        print(f"\nEvaluation completed in {total_time:.1f} seconds")
        print(f"Average time per sample: {total_time/num_samples:.3f} seconds")
        
        # Calculate metrics
        self.calculate_metrics()
        
    def calculate_metrics(self):
        """Calculate detailed performance metrics"""
        print("\n" + "="*80)
        print("PERFORMANCE METRICS")
        print("="*80)
        
        # Convert to numpy arrays
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        confidences = np.array(self.confidences)
        
        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"\n🎯 ENSEMBLE ACCURACY: {accuracy:.2%}")
        
        # Average confidence
        avg_confidence = np.mean(confidences)
        print(f"📊 Average Confidence: {avg_confidence:.2%}")
        
        # Confidence on correct vs incorrect
        correct_mask = y_true == y_pred
        conf_correct = np.mean(confidences[correct_mask]) if np.any(correct_mask) else 0
        conf_incorrect = np.mean(confidences[~correct_mask]) if np.any(~correct_mask) else 0
        print(f"✅ Confidence when correct: {conf_correct:.2%}")
        print(f"❌ Confidence when wrong: {conf_incorrect:.2%}")
        
        # Individual model accuracies
        print("\n" + "-"*40)
        print("INDIVIDUAL MODEL ACCURACIES:")
        print("-"*40)
        
        model_names = {
            'model_1': 'EfficientNet-B1',
            'model_2': 'ResNet-50 CBAM',
            'model_3': 'DenseNet-121',
            'model_4': 'AST',
            'model_5': 'ConvNeXt-Tiny',
            'model_6': 'PANNs CNN14'
        }
        
        model_accuracies = {}
        for model_key, preds in self.model_predictions.items():
            # Filter out error markers (-1)
            valid_mask = np.array(preds) != -1
            if np.any(valid_mask):
                valid_preds = np.array(preds)[valid_mask]
                valid_true = y_true[valid_mask]
                model_acc = accuracy_score(valid_true, valid_preds)
                model_accuracies[model_key] = model_acc
                print(f"  {model_names[model_key]:20s}: {model_acc:.2%}")
            else:
                print(f"  {model_names[model_key]:20s}: Failed")
        
        # Per-class metrics
        print("\n" + "-"*40)
        print("PER-CLASS PERFORMANCE:")
        print("-"*40)
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.ensemble.species_names,
            output_dict=True,
            zero_division=0
        )
        
        # Sort classes by F1-score
        class_scores = []
        for i, species in enumerate(self.ensemble.species_names):
            if species in report:
                metrics = report[species]
                class_scores.append({
                    'species': species,
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1-score'],
                    'support': int(metrics['support'])
                })
        
        class_scores.sort(key=lambda x: x['f1'], reverse=True)
        
        # Show top 5 best
        print("\n✅ Top 5 Best Performing Species:")
        for i, score in enumerate(class_scores[:5], 1):
            print(f"  {i}. {score['species']:25s} "
                  f"F1: {score['f1']:.2%} | "
                  f"Precision: {score['precision']:.2%} | "
                  f"Recall: {score['recall']:.2%}")
        
        # Show bottom 5
        print("\n❌ Top 5 Most Challenging Species:")
        for i, score in enumerate(class_scores[-5:], 1):
            print(f"  {i}. {score['species']:25s} "
                  f"F1: {score['f1']:.2%} | "
                  f"Precision: {score['precision']:.2%} | "
                  f"Recall: {score['recall']:.2%}")
        
        # Store results
        self.results = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'confidence_correct': conf_correct,
            'confidence_incorrect': conf_incorrect,
            'model_accuracies': model_accuracies,
            'per_class_metrics': class_scores,
            'classification_report': report
        }
        
        return self.results
    
    def generate_confusion_matrix(self, save_path='confusion_matrix.png'):
        """Generate and save confusion matrix"""
        print("\nGenerating confusion matrix...")
        
        y_true = np.array(self.true_labels)
        y_pred = np.array(self.predictions)
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(20, 16))
        
        # Plot with annotations
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.ensemble.species_names,
                   yticklabels=self.ensemble.species_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Ensemble Confusion Matrix\nAccuracy: {self.results["accuracy"]:.2%}', 
                 fontsize=16, pad=20)
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save figure
        output_path = Path('outputs') / save_path
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {output_path}")
        plt.close()
        
        # Also save normalized version
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(cm_normalized, annot=True, fmt='.1%', cmap='Blues',
                   xticklabels=self.ensemble.species_names,
                   yticklabels=self.ensemble.species_names,
                   cbar_kws={'label': 'Percentage'})
        
        plt.title(f'Ensemble Confusion Matrix (Normalized)\nAccuracy: {self.results["accuracy"]:.2%}', 
                 fontsize=16, pad=20)
        plt.xlabel('Predicted Species', fontsize=12)
        plt.ylabel('True Species', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        norm_path = output_path.parent / f"normalized_{save_path}"
        plt.savefig(norm_path, dpi=150, bbox_inches='tight')
        print(f"Normalized confusion matrix saved to: {norm_path}")
        plt.close()
    
    def save_results(self, output_dir='outputs'):
        """Save all results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results as JSON
        results_file = output_path / f'ensemble_test_results_{timestamp}.json'
        with open(results_file, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {
                'timestamp': timestamp,
                'num_samples': len(self.predictions),
                'overall_accuracy': float(self.results['accuracy']),
                'avg_confidence': float(self.results['avg_confidence']),
                'confidence_correct': float(self.results['confidence_correct']),
                'confidence_incorrect': float(self.results['confidence_incorrect']),
                'model_accuracies': {k: float(v) for k, v in self.results['model_accuracies'].items()},
                'per_class_metrics': self.results['per_class_metrics']
            }
            json.dump(json_results, f, indent=2)
        print(f"\nResults saved to: {results_file}")
        
        # Save predictions
        predictions_file = output_path / f'predictions_{timestamp}.npz'
        np.savez(predictions_file,
                predictions=self.predictions,
                true_labels=self.true_labels,
                confidences=self.confidences)
        print(f"Predictions saved to: {predictions_file}")
        
        # Generate confusion matrices
        self.generate_confusion_matrix(f'confusion_matrix_{timestamp}.png')
        
        # Generate summary report
        self.generate_summary_report(output_path / f'summary_report_{timestamp}.txt')
    
    def generate_summary_report(self, output_path):
        """Generate a text summary report"""
        with open(output_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENSEMBLE MODEL TEST EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Samples: {len(self.predictions)}\n")
            f.write(f"Number of Classes: {self.ensemble.num_classes}\n")
            f.write(f"Number of Models: {len(self.ensemble.models)}\n\n")
            
            f.write("-"*40 + "\n")
            f.write("OVERALL PERFORMANCE\n")
            f.write("-"*40 + "\n")
            f.write(f"Ensemble Accuracy: {self.results['accuracy']:.2%}\n")
            f.write(f"Average Confidence: {self.results['avg_confidence']:.2%}\n")
            f.write(f"Confidence when Correct: {self.results['confidence_correct']:.2%}\n")
            f.write(f"Confidence when Wrong: {self.results['confidence_incorrect']:.2%}\n\n")
            
            f.write("-"*40 + "\n")
            f.write("INDIVIDUAL MODEL ACCURACIES\n")
            f.write("-"*40 + "\n")
            
            model_names = {
                'model_1': 'EfficientNet-B1',
                'model_2': 'ResNet-50 CBAM',
                'model_3': 'DenseNet-121',
                'model_4': 'AST',
                'model_5': 'ConvNeXt-Tiny',
                'model_6': 'PANNs CNN14'
            }
            
            for model_key, acc in self.results['model_accuracies'].items():
                f.write(f"{model_names[model_key]:25s}: {acc:.2%}\n")
            
            f.write("\n" + "-"*40 + "\n")
            f.write("TOP 10 BEST PERFORMING SPECIES\n")
            f.write("-"*40 + "\n")
            
            for i, score in enumerate(self.results['per_class_metrics'][:10], 1):
                f.write(f"{i:2d}. {score['species']:25s} "
                       f"F1: {score['f1']:.2%} "
                       f"(P: {score['precision']:.2%}, R: {score['recall']:.2%})\n")
            
            f.write("\n" + "-"*40 + "\n")
            f.write("TOP 10 MOST CHALLENGING SPECIES\n")
            f.write("-"*40 + "\n")
            
            for i, score in enumerate(self.results['per_class_metrics'][-10:], 1):
                f.write(f"{i:2d}. {score['species']:25s} "
                       f"F1: {score['f1']:.2%} "
                       f"(P: {score['precision']:.2%}, R: {score['recall']:.2%})\n")
        
        print(f"Summary report saved to: {output_path}")


def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate ensemble model on test set')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation (default: 32)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (default: all)')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Directory to save results (default: outputs)')
    parser.add_argument('--config', type=str, default='configs/ensemble_config.yaml',
                       help='Path to ensemble config (default: configs/ensemble_config.yaml)')
    
    args = parser.parse_args()
    
    print("="*80)
    print("ENSEMBLE MODEL TEST SET EVALUATION")
    print("="*80)
    
    # Create evaluator
    evaluator = EnsembleTestEvaluator(args.config)
    
    # Run evaluation
    evaluator.run_evaluation(
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    # Save results
    evaluator.save_results(args.output_dir)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\n🎯 Final Ensemble Accuracy: {evaluator.results['accuracy']:.2%}")
    print(f"📊 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
