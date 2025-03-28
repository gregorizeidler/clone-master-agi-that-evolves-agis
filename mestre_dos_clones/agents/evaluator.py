from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import time
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from mestre_dos_clones.clones.base_clone import BaseClone
from mestre_dos_clones.clones.sentiment_clone import SentimentClone

# Import additional clone types
try:
    from mestre_dos_clones.clones.regression_clone import RegressionClone
    from mestre_dos_clones.clones.clustering_clone import ClusteringClone
    from mestre_dos_clones.clones.dimension_clone import DimensionClone
    ADDITIONAL_CLONES_AVAILABLE = True
except ImportError:
    ADDITIONAL_CLONES_AVAILABLE = False


class Evaluator:
    """
    Evaluator Agent responsible for testing the performance of clones.
    Evaluates each clone in controlled scenarios and calculates metrics.
    """
    
    def __init__(self):
        """Initializes the Evaluator."""
        self.history = []
        self.test_data = None
        self.evaluation_cache = {}  # Cache of evaluations to avoid unnecessary reassessments
        
    def log_action(self, action_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Records an action by the Evaluator.
        
        Args:
            action_type: Type of action ('evaluate', 'rank', etc)
            details: Action details
            
        Returns:
            Dict: Action record
        """
        log_entry = {
            'agent': 'evaluator',
            'action': action_type,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.history.append(log_entry)
        return log_entry
    
    def set_test_data(self, texts: List[str], labels: List[int]):
        """
        Sets the data for evaluating clones.
        
        Args:
            texts: List of texts for evaluation
            labels: List of labels (0=negative, 1=positive)
        """
        self.test_data = (texts, labels)
        self.log_action('set_test_data', {
            'num_samples': len(texts),
            'positive_ratio': sum(labels) / len(labels) if len(labels) > 0 else 0
        })
    
    def evaluate_clone(self, clone: BaseClone) -> Dict[str, Any]:
        """
        Evaluates a clone with test data.
        
        Args:
            clone: Clone to be evaluated
            
        Returns:
            Dict: Evaluation metrics
        """
        # Check if we already have test data
        if self.test_data is None:
            raise ValueError("Test data not defined. Use set_test_data() first.")
            
        # Check if we already evaluated this clone previously (same generation)
        cache_key = f"{clone.id}-{clone.training_time if hasattr(clone, 'training_time') else id(clone)}"
        if cache_key in self.evaluation_cache:
            return self.evaluation_cache[cache_key]
            
        # Extract test data
        features, true_values = self.test_data
        
        # Evaluation start time
        start_time = time.time()
        
        # Different metrics depending on clone type
        if isinstance(clone, SentimentClone):
            # Get predictions from sentiment clone
            predictions = clone.predict(features)
            
            # Calculate classification metrics
            metrics = {
                'accuracy': accuracy_score(true_values, predictions),
                'precision': precision_score(true_values, predictions, zero_division=0),
                'recall': recall_score(true_values, predictions, zero_division=0),
                'f1': f1_score(true_values, predictions, zero_division=0),
                'evaluation_time': time.time() - start_time,
                'samples': len(features)
            }
            
            # Calculate confusion matrix
            cm = confusion_matrix(true_values, predictions)
            metrics['confusion_matrix'] = cm.tolist()
            
            # Update clone score
            clone.performance_score = metrics['f1']  # We use F1 as the main metric
            
        elif ADDITIONAL_CLONES_AVAILABLE and isinstance(clone, RegressionClone):
            # Get predictions from regression clone
            predictions = clone.predict(features)
            
            # Calculate regression metrics
            metrics = {
                'mse': mean_squared_error(true_values, predictions),
                'mae': mean_absolute_error(true_values, predictions),
                'r2': r2_score(true_values, predictions),
                'evaluation_time': time.time() - start_time,
                'samples': len(features)
            }
            
            # Update clone score
            # For regression, we use a combination of metrics (higher is better)
            clone.performance_score = clone.evaluate(features, true_values)
            
        elif ADDITIONAL_CLONES_AVAILABLE and isinstance(clone, ClusteringClone):
            # For clustering, we use the internal evaluation method
            clone.performance_score = clone.evaluate(features)
            
            metrics = {
                'cluster_score': clone.performance_score,
                'evaluation_time': time.time() - start_time,
                'samples': len(features)
            }
            
        elif ADDITIONAL_CLONES_AVAILABLE and isinstance(clone, DimensionClone):
            # For dimensionality reduction, we use the internal evaluation method
            clone.performance_score = clone.evaluate(features)
            
            metrics = {
                'dimension_score': clone.performance_score,
                'evaluation_time': time.time() - start_time,
                'samples': len(features)
            }
            
        else:
            # For other unrecognized clone types
            # We try to use the generic evaluate method of the clone
            try:
                predictions = clone.predict(features)
                clone.performance_score = clone.evaluate(features, true_values)
                
                metrics = {
                    'generic_score': clone.performance_score,
                    'evaluation_time': time.time() - start_time,
                    'samples': len(features)
                }
            except Exception as e:
                # If it fails, we record the error and return a low score
                metrics = {
                    'error': str(e),
                    'evaluation_time': time.time() - start_time,
                    'samples': len(features)
                }
                clone.performance_score = 0.01  # Low score but not zero
        
        # Store in cache
        self.evaluation_cache[cache_key] = metrics
        
        # Record evaluation
        self.log_action('evaluate', {
            'clone_id': clone.id,
            'clone_name': clone.name,
            'metrics': metrics
        })
        
        return metrics
    
    def evaluate_population(self, clones: List[BaseClone]) -> Dict[str, Dict[str, Any]]:
        """
        Evaluates an entire population of clones.
        
        Args:
            clones: List of clones to be evaluated
            
        Returns:
            Dict: Dictionary with evaluation metrics per clone
        """
        results = {}
        for clone in clones:
            clone_metrics = self.evaluate_clone(clone)
            results[clone.id] = {
                'name': clone.name,
                'metrics': clone_metrics,
                'score': clone.performance_score
            }
        
        # Record population evaluation
        self.log_action('evaluate_population', {
            'num_clones': len(clones),
            'clone_ids': [clone.id for clone in clones]
        })
        
        return results
    
    def rank_clones(self, clones: List[BaseClone], metric: str = None) -> List[Tuple[BaseClone, float]]:
        """
        Ranks the clones based on their performance.
        
        Args:
            clones: List of clones to be ranked
            metric: Metric to be used for ranking, if None uses performance_score
            
        Returns:
            List[Tuple[BaseClone, float]]: Ordered list of clones and their scores
        """
        # Ensure all clones have been evaluated
        for clone in clones:
            if clone.performance_score is None:
                self.evaluate_clone(clone)
        
        # If we don't have a specific metric, use the performance score
        if metric is None:
            clone_scores = [(clone, clone.performance_score) for clone in clones]
        else:
            # For specific metrics, we need to get from the evaluation results
            clone_scores = []
            for clone in clones:
                cache_key = f"{clone.id}-{clone.training_time if hasattr(clone, 'training_time') else id(clone)}"
                if cache_key in self.evaluation_cache:
                    metrics = self.evaluation_cache[cache_key]
                    score = metrics.get(metric, 0.0)
                    clone_scores.append((clone, score))
                else:
                    metrics = self.evaluate_clone(clone)
                    score = metrics.get(metric, 0.0)
                    clone_scores.append((clone, score))
        
        # Sort by score (descending)
        ranked_clones = sorted(clone_scores, key=lambda x: x[1], reverse=True)
        
        # Record the ranking
        self.log_action('rank', {
            'metric': metric,
            'num_clones': len(clones),
            'top_clone': {
                'id': ranked_clones[0][0].id,
                'name': ranked_clones[0][0].name,
                'score': ranked_clones[0][1]
            } if ranked_clones else None
        })
        
        return ranked_clones
    
    def save_evaluation_results(self, results: Dict[str, Dict[str, Any]], 
                               filepath: str = 'evaluation_results.json'):
        """
        Saves the evaluation results to a file.
        
        Args:
            results: Evaluation results
            filepath: Path to save the results
        """
        # Convert to serializable format
        serializable_results = {}
        for clone_id, data in results.items():
            # Convert numpy arrays to lists
            metrics = {}
            for k, v in data['metrics'].items():
                if isinstance(v, (np.ndarray, np.number)):
                    metrics[k] = v.tolist() if hasattr(v, 'tolist') else float(v)
                else:
                    metrics[k] = v
            
            serializable_results[clone_id] = {
                'name': data['name'],
                'metrics': metrics,
                'score': float(data['score']) if 'score' in data else None
            }
        
        # Save to file
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        self.log_action('save_results', {
            'filepath': filepath,
            'num_clones': len(results)
        })
        
        return filepath
