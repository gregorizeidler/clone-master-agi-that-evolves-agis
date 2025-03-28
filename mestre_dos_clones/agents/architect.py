import random
import numpy as np
from typing import Dict, List, Any, Optional, Type

from mestre_dos_clones.clones.sentiment_clone import SentimentClone
from mestre_dos_clones.clones.adaptive_clone import AdaptiveClone
from mestre_dos_clones.clones.meta_clone import MetaClone
from mestre_dos_clones.clones.base_clone import BaseClone

# Import new clone types
from mestre_dos_clones.clones.regression_clone import RegressionClone
from mestre_dos_clones.clones.clustering_clone import ClusteringClone
from mestre_dos_clones.clones.dimension_clone import DimensionClone


class Architect:
    """
    Architect Agent responsible for designing new clones.
    Creates the initial structure or modifies existing clones.
    """
    
    def __init__(self):
        """Initializes the Architect."""
        self.created_clones = 0
        self.history = []
        # Define the types of clones available for the architect to create
        self.available_clone_types = [SentimentClone]
        # Weights for choosing clone types (used when there are multiple types)
        self.clone_type_weights = [1.0]
        
    def log_action(self, action_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Records an action by the Architect.
        
        Args:
            action_type: Type of action ('create', 'modify', etc)
            details: Action details
            
        Returns:
            Dict: Action record
        """
        log_entry = {
            'agent': 'architect',
            'action': action_type,
            'details': details or {}
        }
        self.history.append(log_entry)
        return log_entry
    
    def create_sentiment_clone(self, strategy: str = 'random') -> SentimentClone:
        """
        Creates a new clone for sentiment analysis.
        
        Args:
            strategy: Creation strategy ('random', 'naive_bayes', 'logistic', 'svm')
            
        Returns:
            SentimentClone: A new clone
        """
        self.created_clones += 1
        
        if strategy == 'naive_bayes':
            algorithm = 'naive_bayes'
            hyperparams = {'alpha': random.uniform(0.1, 2.0)}
        elif strategy == 'logistic':
            algorithm = 'logistic'
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'max_iter': random.randint(100, 500)
            }
        elif strategy == 'svm':
            algorithm = 'svm'
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'kernel': random.choice(['linear', 'rbf'])
            }
        elif strategy == 'random_forest':
            algorithm = 'random_forest'
            hyperparams = {
                'n_estimators': random.randint(50, 200),
                'max_depth': random.choice([None, 5, 10, 15, 20]),
                'min_samples_split': random.randint(2, 10)
            }
        else:  # random
            algorithm = random.choice(['naive_bayes', 'logistic', 'svm', 'random_forest'])
            
            if algorithm == 'naive_bayes':
                hyperparams = {'alpha': random.uniform(0.1, 2.0)}
            elif algorithm == 'logistic':
                hyperparams = {
                    'C': random.uniform(0.1, 10.0),
                    'max_iter': random.randint(100, 500)
                }
            elif algorithm == 'svm':
                hyperparams = {
                    'C': random.uniform(0.1, 10.0),
                    'kernel': random.choice(['linear', 'rbf'])
                }
            else:  # random_forest
                hyperparams = {
                    'n_estimators': random.randint(50, 200),
                    'max_depth': random.choice([None, 5, 10, 15, 20]),
                    'min_samples_split': random.randint(2, 10)
                }
        
        # Choose the vectorizer type
        vectorizer = random.choice(['count', 'tfidf'])
        
        # Create the clone
        clone = SentimentClone(
            algorithm=algorithm,
            vectorizer=vectorizer,
            hyperparams=hyperparams,
            name=f"Genesis-{self.created_clones}"
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'SentimentClone',
            'algorithm': algorithm,
            'vectorizer': vectorizer,
            'hyperparams': hyperparams
        })
        
        return clone
    
    def create_adaptive_clone(self, strategy: str = 'random') -> AdaptiveClone:
        """
        Creates a new adaptive clone with self-adaptation capabilities.
        
        Args:
            strategy: Creation strategy ('random', 'naive_bayes', 'logistic', 'svm', 'random_forest')
            
        Returns:
            AdaptiveClone: A new adaptive clone
        """
        self.created_clones += 1
        
        # Reuse logic to generate algorithm and hyperparameters
        if strategy == 'random':
            algorithm = random.choice(['naive_bayes', 'logistic', 'svm', 'random_forest'])
        else:
            algorithm = strategy
            
        if algorithm == 'naive_bayes':
            hyperparams = {'alpha': random.uniform(0.1, 2.0)}
        elif algorithm == 'logistic':
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'max_iter': random.randint(100, 500)
            }
        elif algorithm == 'svm':
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'kernel': random.choice(['linear', 'rbf'])
            }
        else:  # random_forest
            hyperparams = {
                'n_estimators': random.randint(50, 200),
                'max_depth': random.choice([None, 5, 10, 15, 20]),
                'min_samples_split': random.randint(2, 10)
            }
        
        # Choose the vectorizer type
        vectorizer = random.choice(['count', 'tfidf'])
        
        # Generate initial mutation rates
        mutation_rates = {
            'algorithm_change_rate': random.uniform(0.1, 0.3),
            'vectorizer_change_rate': random.uniform(0.2, 0.4),
            'param_mutation_strength': random.uniform(0.2, 0.4),
            'param_mutation_rate': random.uniform(0.3, 0.7)
        }
        
        # Create the adaptive clone
        clone = AdaptiveClone(
            algorithm=algorithm,
            vectorizer=vectorizer,
            hyperparams=hyperparams,
            mutation_rates=mutation_rates,
            name=f"Adaptive-{self.created_clones}"
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'AdaptiveClone',
            'algorithm': algorithm,
            'vectorizer': vectorizer,
            'hyperparams': hyperparams,
            'mutation_rates': mutation_rates
        })
        
        return clone
    
    def create_meta_clone(self, strategy: str = 'random') -> MetaClone:
        """
        Creates a new clone with meta-learning capabilities.
        
        Args:
            strategy: Creation strategy ('random', 'naive_bayes', 'logistic', 'svm', 'random_forest')
            
        Returns:
            MetaClone: A new clone with meta-learning
        """
        self.created_clones += 1
        
        # Reuse logic to generate algorithm and hyperparameters
        if strategy == 'random':
            algorithm = random.choice(['naive_bayes', 'logistic', 'svm', 'random_forest'])
        else:
            algorithm = strategy
            
        if algorithm == 'naive_bayes':
            hyperparams = {'alpha': random.uniform(0.1, 2.0)}
        elif algorithm == 'logistic':
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'max_iter': random.randint(100, 500)
            }
        elif algorithm == 'svm':
            hyperparams = {
                'C': random.uniform(0.1, 10.0),
                'kernel': random.choice(['linear', 'rbf'])
            }
        else:  # random_forest
            hyperparams = {
                'n_estimators': random.randint(50, 200),
                'max_depth': random.choice([None, 5, 10, 15, 20]),
                'min_samples_split': random.randint(2, 10)
            }
        
        # Choose the vectorizer type
        vectorizer = random.choice(['count', 'tfidf'])
        
        # Generate initial mutation rates
        mutation_rates = {
            'algorithm_change_rate': random.uniform(0.1, 0.3),
            'vectorizer_change_rate': random.uniform(0.2, 0.4),
            'param_mutation_strength': random.uniform(0.2, 0.4),
            'param_mutation_rate': random.uniform(0.3, 0.7)
        }
        
        # Generate meta-strategies with some randomly activated
        meta_strategies = {
            'feature_selection': {
                'active': random.random() < 0.3,  # 30% chance of being active
                'threshold': random.uniform(0.005, 0.02),
                'min_features': random.randint(50, 200),
                'max_features': random.randint(1000, 10000)
            },
            'error_analysis': {
                'active': random.random() < 0.3,
                'sample_size': random.uniform(0.1, 0.3),
                'error_threshold': random.uniform(0.2, 0.4)
            },
            'learning_rate_adjustment': {
                'active': random.random() < 0.3,
                'initial_lr': random.uniform(0.05, 0.2),
                'decay_factor': random.uniform(0.8, 0.95),
                'min_lr': random.uniform(0.0005, 0.002)
            },
            'early_stopping': {
                'active': random.random() < 0.3,
                'patience': random.randint(2, 5),
                'min_delta': random.uniform(0.0005, 0.005)
            },
            'sample_weighting': {
                'active': random.random() < 0.3,
                'weight_errors': random.uniform(1.2, 2.0),
                'reweight_interval': random.randint(1, 3)
            }
        }
        
        # Create the meta-learning clone
        clone = MetaClone(
            algorithm=algorithm,
            vectorizer=vectorizer,
            hyperparams=hyperparams,
            mutation_rates=mutation_rates,
            meta_strategies=meta_strategies,
            name=f"Meta-{self.created_clones}"
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'MetaClone',
            'algorithm': algorithm,
            'vectorizer': vectorizer,
            'hyperparams': hyperparams,
            'mutation_rates': mutation_rates,
            'meta_strategies': {k: {'active': v['active']} for k, v in meta_strategies.items()}
        })
        
        return clone
    
    def create_regression_clone(self, strategy: str = 'random') -> RegressionClone:
        """
        Creates a clone for regression.
        
        Args:
            strategy: Creation strategy ('random' or specific algorithm)
            
        Returns:
            RegressionClone: A new regression clone
        """
        self.created_clones += 1
        
        # Choose a random regression algorithm
        if strategy == 'random':
            algorithm = random.choice(list(RegressionClone.AVAILABLE_ALGORITHMS.keys()))
        else:
            algorithm = strategy if strategy in RegressionClone.AVAILABLE_ALGORITHMS else 'random_forest_regressor'
        
        # Create the clone
        clone = RegressionClone(
            algorithm_name=algorithm,
            name=f"Regression-{self.created_clones}",
            seed=random.randint(1, 10000)
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'RegressionClone',
            'algorithm': algorithm
        })
        
        return clone
    
    def create_clustering_clone(self, strategy: str = 'random') -> ClusteringClone:
        """
        Creates a clone for clustering.
        
        Args:
            strategy: Creation strategy ('random' or specific algorithm)
            
        Returns:
            ClusteringClone: A new clustering clone
        """
        self.created_clones += 1
        
        # Choose a random clustering algorithm
        if strategy == 'random':
            algorithm = random.choice(list(ClusteringClone.AVAILABLE_ALGORITHMS.keys()))
        else:
            algorithm = strategy if strategy in ClusteringClone.AVAILABLE_ALGORITHMS else 'kmeans'
        
        # Choose a random number of clusters
        n_clusters = random.randint(2, 8)
        
        # Create the clone
        clone = ClusteringClone(
            algorithm_name=algorithm,
            n_clusters=n_clusters,
            name=f"Cluster-{self.created_clones}",
            seed=random.randint(1, 10000)
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'ClusteringClone',
            'algorithm': algorithm,
            'n_clusters': n_clusters
        })
        
        return clone
    
    def create_dimension_clone(self, strategy: str = 'random') -> DimensionClone:
        """
        Creates a clone for dimensionality reduction.
        
        Args:
            strategy: Creation strategy ('random' or specific algorithm)
            
        Returns:
            DimensionClone: A new clone for dimensionality reduction
        """
        self.created_clones += 1
        
        # Choose a random dimensionality reduction algorithm
        if strategy == 'random':
            algorithm = random.choice(list(DimensionClone.AVAILABLE_ALGORITHMS.keys()))
        else:
            algorithm = strategy if strategy in DimensionClone.AVAILABLE_ALGORITHMS else 'pca'
        
        # Choose a random number of components
        n_components = random.randint(2, 5)
        
        # Create the clone
        clone = DimensionClone(
            algorithm_name=algorithm,
            n_components=n_components,
            name=f"Dimension-{self.created_clones}",
            seed=random.randint(1, 10000)
        )
        
        # Record the creation
        self.log_action('create', {
            'clone_id': clone.id,
            'clone_type': 'DimensionClone',
            'algorithm': algorithm,
            'n_components': n_components
        })
        
        return clone
    
    def create_clone(self, clone_type: Type = None, strategy: str = 'random') -> BaseClone:
        """
        Creates a new clone of the specified type.
        
        Args:
            clone_type: Type of clone to create (class)
            strategy: Creation strategy
            
        Returns:
            BaseClone: A new clone of the specified type
        """
        # If not specified, choose one of the available types
        if clone_type is None:
            if len(self.available_clone_types) == 1:
                clone_type = self.available_clone_types[0]
            else:
                # Choose a type based on weights
                clone_type = random.choices(
                    self.available_clone_types, 
                    weights=self.clone_type_weights[:len(self.available_clone_types)],
                    k=1
                )[0]
        
        # Create the clone according to the type
        if clone_type == SentimentClone:
            return self.create_sentiment_clone(strategy)
        elif clone_type == AdaptiveClone:
            return self.create_adaptive_clone(strategy)
        elif clone_type == MetaClone:
            return self.create_meta_clone(strategy)
        elif clone_type == RegressionClone:
            return self.create_regression_clone(strategy)
        elif clone_type == ClusteringClone:
            return self.create_clustering_clone(strategy)
        elif clone_type == DimensionClone:
            return self.create_dimension_clone(strategy)
        else:
            # Unknown type, use default SentimentClone
            return self.create_sentiment_clone(strategy)
    
    def modify_clone(self, clone: BaseClone, modification_type: str = 'mutate') -> BaseClone:
        """
        Modifies an existing clone.
        
        Args:
            clone: Clone to modify
            modification_type: Type of modification ('mutate', 'optimize', 'evolve')
            
        Returns:
            BaseClone: Modified clone
        """
        # For AdaptiveClone and MetaClone, we can simply use the mutate method
        # which already implements self-adaptation
        if isinstance(clone, (AdaptiveClone, MetaClone)):
            new_clone = clone.mutate()
            
            # Record the modification
            self.log_action('modify', {
                'original_clone_id': clone.id,
                'new_clone_id': new_clone.id,
                'clone_type': new_clone.__class__.__name__,
                'modification_type': 'mutate'
            })
            
            return new_clone
        
        # For SentimentClone or other types, use existing logic
        if modification_type == 'mutate':
            # Simple random mutation
            new_clone = clone.mutate()
        
        elif modification_type == 'optimize':
            # Small optimizations to current hyperparameters
            hyperparams = dict(clone.hyperparams)
            
            if clone.algorithm_name == 'naive_bayes':
                # Optimize Naive Bayes alpha
                current_alpha = hyperparams.get('alpha', 1.0)
                hyperparams['alpha'] = max(0.01, current_alpha * random.uniform(0.8, 1.2))
                
            elif clone.algorithm_name == 'logistic':
                # Optimize logistic regression C
                current_c = hyperparams.get('C', 1.0)
                hyperparams['C'] = max(0.1, current_c * random.uniform(0.8, 1.2))
                
            elif clone.algorithm_name == 'svm':
                # Optimize SVM C
                current_c = hyperparams.get('C', 1.0)
                hyperparams['C'] = max(0.1, current_c * random.uniform(0.8, 1.2))
            
            elif clone.algorithm_name == 'random_forest':
                # Optimize random forest n_estimators
                current_n = hyperparams.get('n_estimators', 100)
                hyperparams['n_estimators'] = max(10, int(current_n * random.uniform(0.8, 1.2)))
            
            # Create new clone with optimized parameters
            new_clone = SentimentClone(
                algorithm=clone.algorithm_name,
                vectorizer=clone.vectorizer_name,
                hyperparams=hyperparams,
                name=f"Optimized-{clone.id}-{str(random.randint(1000, 9999))}"
            )
            
            # Configure parent relationship
            new_clone.parent_ids = [clone.id]
            new_clone.generation = clone.generation + 1
            
        elif modification_type == 'evolve':
            # More significant evolution (can change algorithm/vectorizer)
            algorithm_options = ['naive_bayes', 'logistic', 'svm', 'random_forest']
            vectorizer_options = ['count', 'tfidf']
            
            # 50% chance to change algorithm
            new_algorithm = random.choice(algorithm_options) if random.random() < 0.5 else clone.algorithm_name
            
            # 50% chance to change vectorizer
            new_vectorizer = random.choice(vectorizer_options) if random.random() < 0.5 else clone.vectorizer_name
            
            # New appropriate hyperparameters for the algorithm
            if new_algorithm == 'naive_bayes':
                hyperparams = {'alpha': random.uniform(0.1, 2.0)}
            elif new_algorithm == 'logistic':
                hyperparams = {
                    'C': random.uniform(0.1, 10.0),
                    'max_iter': random.randint(100, 500)
                }
            elif new_algorithm == 'svm':
                hyperparams = {
                    'C': random.uniform(0.1, 10.0),
                    'kernel': random.choice(['linear', 'rbf'])
                }
            else:  # random_forest
                hyperparams = {
                    'n_estimators': random.randint(50, 200),
                    'max_depth': random.choice([None, 5, 10, 15, 20]),
                    'min_samples_split': random.randint(2, 10)
                }
            
            # Determine the type of the new clone
            # 70% chance to keep the same type, 30% chance to evolve to a more advanced type
            if random.random() < 0.7:
                clone_class = clone.__class__
            else:
                # Choose a more advanced class
                if isinstance(clone, SentimentClone) and not isinstance(clone, AdaptiveClone):
                    clone_class = random.choice([AdaptiveClone, MetaClone])
                elif isinstance(clone, AdaptiveClone) and not isinstance(clone, MetaClone):
                    clone_class = MetaClone
                else:
                    clone_class = clone.__class__
            
            # Create basic arguments
            kwargs = {
                'algorithm': new_algorithm,
                'vectorizer': new_vectorizer,
                'hyperparams': hyperparams,
                'name': f"Evolved-{clone.id}-{str(random.randint(1000, 9999))}"
            }
            
            # Add type-specific arguments
            if clone_class in [AdaptiveClone, MetaClone]:
                kwargs['mutation_rates'] = {
                    'algorithm_change_rate': random.uniform(0.1, 0.3),
                    'vectorizer_change_rate': random.uniform(0.2, 0.4),
                    'param_mutation_strength': random.uniform(0.2, 0.4),
                    'param_mutation_rate': random.uniform(0.3, 0.7)
                }
            
            if clone_class == MetaClone:
                kwargs['meta_strategies'] = {
                    'feature_selection': {
                        'active': random.random() < 0.5,
                        'threshold': random.uniform(0.005, 0.02),
                        'min_features': random.randint(50, 200),
                        'max_features': random.randint(1000, 10000)
                    },
                    'error_analysis': {
                        'active': random.random() < 0.5,
                        'sample_size': random.uniform(0.1, 0.3),
                        'error_threshold': random.uniform(0.2, 0.4)
                    },
                    'learning_rate_adjustment': {
                        'active': random.random() < 0.5,
                        'initial_lr': random.uniform(0.05, 0.2),
                        'decay_factor': random.uniform(0.8, 0.95),
                        'min_lr': random.uniform(0.0005, 0.002)
                    },
                    'early_stopping': {
                        'active': random.random() < 0.5,
                        'patience': random.randint(2, 5),
                        'min_delta': random.uniform(0.0005, 0.005)
                    },
                    'sample_weighting': {
                        'active': random.random() < 0.5,
                        'weight_errors': random.uniform(1.2, 2.0),
                        'reweight_interval': random.randint(1, 3)
                    }
                }
            
            # Create the evolved clone
            new_clone = clone_class(**kwargs)
            
            # Configure parent relationship
            new_clone.parent_ids = [clone.id]
            new_clone.generation = clone.generation + 1
            
        else:
            # Default case, do a simple mutation
            new_clone = clone.mutate()
        
        # Record the modification
        self.log_action('modify', {
            'original_clone_id': clone.id,
            'new_clone_id': new_clone.id,
            'original_type': clone.__class__.__name__,
            'new_type': new_clone.__class__.__name__,
            'modification_type': modification_type,
            'changes': {
                'algorithm': new_clone.algorithm_name if new_clone.algorithm_name != clone.algorithm_name else 'unchanged',
                'vectorizer': new_clone.vectorizer_name if new_clone.vectorizer_name != clone.vectorizer_name else 'unchanged'
            }
        })
        
        return new_clone
    
    def create_initial_population(self, size: int = 5) -> List[BaseClone]:
        """
        Creates an initial population of various clones.
        
        Args:
            size: Number of clones to create
            
        Returns:
            List[BaseClone]: List of created clones
        """
        population = []
        
        # If we only have one type of clone, create all of that type
        if len(self.available_clone_types) == 1:
            clone_type = self.available_clone_types[0]
            
            # Special check for clones that don't use the clone_type parameter
            if clone_type in [RegressionClone, ClusteringClone, DimensionClone]:
                # For specialized clones, call the specific creation method
                if clone_type == RegressionClone:
                    # Ensure we have at least one of each main algorithm
                    strategies = ['linear_regression', 'ridge', 'random_forest_regressor']
                    for strategy in strategies[:min(size, len(strategies))]:
                        population.append(self.create_regression_clone(strategy))
                    
                    # Complete the population with random clones
                    remaining = size - len(population)
                    for _ in range(remaining):
                        population.append(self.create_regression_clone('random'))
                
                elif clone_type == ClusteringClone:
                    # Ensure we have at least one of each main algorithm
                    strategies = ['kmeans', 'dbscan', 'hierarchical']
                    for strategy in strategies[:min(size, len(strategies))]:
                        population.append(self.create_clustering_clone(strategy))
                    
                    # Complete the population with random clones
                    remaining = size - len(population)
                    for _ in range(remaining):
                        population.append(self.create_clustering_clone('random'))
                
                elif clone_type == DimensionClone:
                    # Ensure we have at least one of each main algorithm
                    strategies = ['pca', 'tsne', 'umap']
                    for strategy in strategies[:min(size, len(strategies))]:
                        population.append(self.create_dimension_clone(strategy))
                    
                    # Complete the population with random clones
                    remaining = size - len(population)
                    for _ in range(remaining):
                        population.append(self.create_dimension_clone('random'))
            else:
                # For traditional clone types like SentimentClone
                # Ensure we have at least one of each main algorithm
                strategies = ['naive_bayes', 'logistic', 'svm']
                for strategy in strategies[:min(size, len(strategies))]:
                    population.append(self.create_clone(clone_type, strategy))
                
                # Complete the population with random clones
                remaining = size - len(population)
                for _ in range(remaining):
                    population.append(self.create_clone(clone_type, 'random'))
                
        else:
            # Approximate distribution based on weights
            counts = {}
            total_weight = sum(self.clone_type_weights[:len(self.available_clone_types)])
            
            for i, clone_type in enumerate(self.available_clone_types):
                weight = self.clone_type_weights[i]
                type_count = max(1, int(round((weight / total_weight) * size)))
                counts[clone_type] = min(type_count, size - sum(counts.values()))
                
                # If we've reached the total size, break
                if sum(counts.values()) >= size:
                    break
                    
            # Adjust to ensure exact population size
            while sum(counts.values()) > size:
                for clone_type in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
                    if counts[clone_type] > 1:
                        counts[clone_type] -= 1
                        break
                        
            while sum(counts.values()) < size:
                for clone_type in sorted(counts.keys(), key=lambda x: counts[x]):
                    counts[clone_type] += 1
                    break
            
            # Create clones of each type
            for clone_type, count in counts.items():
                for _ in range(count):
                    population.append(self.create_clone(clone_type, 'random'))
        
        # Record the creation of the population
        self.log_action('create_population', {
            'size': size,
            'clone_ids': [clone.id for clone in population],
            'type_distribution': {t.__name__: sum(1 for c in population if isinstance(c, t)) 
                                  for t in self.available_clone_types}
        })
        
        return population
