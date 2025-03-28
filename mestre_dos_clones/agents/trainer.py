import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sklearn.model_selection import train_test_split

from mestre_dos_clones.clones.base_clone import BaseClone
from mestre_dos_clones.clones.sentiment_clone import SentimentClone


class Trainer:
    """
    Trainer Agent responsible for training clones with data.
    Prepares data, executes training, and validates results.
    """
    
    def __init__(self, data_source=None):
        """
        Initializes the Trainer.
        
        Args:
            data_source: Optional, data source for training
        """
        self.data_source = data_source
        self.history = []
        self.cached_data = None
        
    def log_action(self, action_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Records an action by the Trainer.
        
        Args:
            action_type: Type of action ('train', 'validate', etc)
            details: Action details
            
        Returns:
            Dict: Action record
        """
        log_entry = {
            'agent': 'trainer',
            'action': action_type,
            'details': details or {}
        }
        self.history.append(log_entry)
        return log_entry
    
    def load_data(self, filepath: Optional[str] = None) -> Tuple[List[str], List[int]]:
        """
        Loads data for training.
        
        Args:
            filepath: Path to data file (optional)
            
        Returns:
            Tuple[List[str], List[int]]: Texts and labels (0=negative, 1=positive)
        """
        # If we already have cached data, return it
        if self.cached_data is not None:
            return self.cached_data
        
        if filepath:
            # Load data from a file
            texts = []
            labels = []
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) == 2:
                            text, label = parts
                            texts.append(text)
                            labels.append(int(label))
            except Exception as e:
                self.log_action('load_data_error', {'filepath': filepath, 'error': str(e)})
                # In case of error, generate synthetic data
                texts, labels = self._generate_synthetic_data()
        else:
            # Generate synthetic data if no file was provided
            texts, labels = self._generate_synthetic_data()
        
        # Store in cache for future use
        self.cached_data = (texts, labels)
        
        # Record the loading
        self.log_action('load_data', {
            'source': filepath or 'synthetic',
            'samples': len(texts),
            'positive_ratio': sum(labels) / len(labels) if labels else 0
        })
        
        return texts, labels
    
    def _generate_synthetic_data(self, num_samples: int = 1000) -> Tuple[List[str], List[int]]:
        """
        Generates synthetic data to train clones.
        
        Args:
            num_samples: Number of samples to be generated
            
        Returns:
            Tuple[List[str], List[int]]: Synthetic texts and labels
        """
        texts = []
        labels = []
        
        # Positive and negative words for data generation
        positive_words = [
            "excellent", "great", "good", "wonderful", "amazing", 
            "fantastic", "loved it", "loved", "liked", "recommend",
            "happy", "cheerful", "satisfied", "content", "pleasant"
        ]
        
        negative_words = [
            "terrible", "bad", "awful", "horrible", "hated it",
            "hated", "didn't like", "disappointing", "disappointment", "disappointed",
            "sad", "irritated", "upset", "unsatisfied", "unpleasant"
        ]
        
        # List of neutral words to compose the sentences
        neutral_words = [
            "the", "a", "one", "was", "is", "this", "that", "it",
            "which", "of", "for", "with", "without", "in", "on", "at",
            "product", "service", "experience", "customer service", "quality",
            "price", "value", "time", "day", "today", "yesterday", "always",
            "never", "very", "little", "quite", "really", "simply"
        ]
        
        for _ in range(num_samples):
            # Decide if the text will be positive or negative
            is_positive = random.random() > 0.5
            label = 1 if is_positive else 0
            
            # List of words to compose the text
            sentiment_words = positive_words if is_positive else negative_words
            
            # Random sentence length (between 5 and 15 words)
            length = random.randint(5, 15)
            
            # Choose words to compose the sentence
            text_words = []
            for i in range(length):
                # 30% chance of using a sentiment word
                if random.random() < 0.3:
                    text_words.append(random.choice(sentiment_words))
                else:
                    text_words.append(random.choice(neutral_words))
            
            # Assemble the final text
            text = ' '.join(text_words)
            # First letter capitalized and period at the end
            text = text[0].upper() + text[1:] + '.'
            
            texts.append(text)
            labels.append(label)
        
        return texts, labels
    
    def prepare_data_splits(self, texts: List[str], labels: List[int], 
                            test_size: float = 0.3, validation_size: float = 0.1,
                            random_state: int = 42) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Divides the data into training, validation, and test sets.
        
        Args:
            texts: List of texts
            labels: List of labels
            test_size: Fraction for test set
            validation_size: Fraction for validation set
            random_state: Seed for reproducibility
            
        Returns:
            Dict: Dictionary with the data sets
        """
        # First, split between training and test
        texts_train, texts_test, labels_train, labels_test = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # If we need a validation set, split again
        if validation_size > 0:
            # Adjust the validation set size proportionally
            valid_size_adjusted = validation_size / (1 - test_size)
            
            texts_train, texts_valid, labels_train, labels_valid = train_test_split(
                texts_train, labels_train, test_size=valid_size_adjusted, 
                random_state=random_state, stratify=labels_train
            )
            
            data_splits = {
                'train': (texts_train, labels_train),
                'validation': (texts_valid, labels_valid),
                'test': (texts_test, labels_test)
            }
        else:
            data_splits = {
                'train': (texts_train, labels_train),
                'test': (texts_test, labels_test)
            }
        
        # Record the data split
        split_stats = {
            'train_size': len(texts_train),
            'test_size': len(texts_test)
        }
        
        if validation_size > 0:
            split_stats['validation_size'] = len(texts_valid)
            
        self.log_action('prepare_splits', split_stats)
        
        return data_splits
    
    def train_clone(self, clone: SentimentClone, texts_train: List[str], labels_train: List[int]) -> Dict[str, Any]:
        """
        Trains a clone with the provided data.
        
        Args:
            clone: Clone to be trained
            texts_train: Training texts
            labels_train: Training labels
            
        Returns:
            Dict: Training metrics
        """
        # Train the clone
        training_metrics = clone.train(texts_train, labels_train)
        
        # Record the training
        self.log_action('train', {
            'clone_id': clone.id,
            'clone_type': clone.__class__.__name__,
            'train_samples': len(texts_train),
            'metrics': training_metrics
        })
        
        return training_metrics
    
    def train_population(self, clones: List[BaseClone], 
                         data_filepath: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """
        Trains an entire population of clones.
        
        Args:
            clones: List of clones to be trained
            data_filepath: Optional path to data file
            
        Returns:
            Dict: Dictionary with training metrics for each clone
        """
        # Load data if not previously loaded
        if self.cached_data is None:
            self.load_data(data_filepath)
            
        texts, labels = self.cached_data
        
        # Prepare data splits
        data_splits = self.prepare_data_splits(texts, labels)
        
        # Train each clone
        results = {}
        for clone in clones:
            # Skip already trained clones
            if any(event.get('action') == 'train' for event in clone.history):
                continue
                
            # Train the clone
            metrics = self.train_clone(clone, data_splits['train'][0], data_splits['train'][1])
            
            # Store results
            results[clone.id] = {
                'name': clone.name,
                'metrics': metrics
            }
        
        return results
