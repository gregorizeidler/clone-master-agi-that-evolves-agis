#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clone Master - Self-evolving multi-agent AGI system
This is the main file that executes the complete evolutionary cycle.
"""

import os
import argparse
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

# Import system agents
from mestre_dos_clones.agents.architect import Architect
from mestre_dos_clones.agents.trainer import Trainer
from mestre_dos_clones.agents.evaluator import Evaluator
from mestre_dos_clones.agents.selector import Selector

# Import clones
from mestre_dos_clones.clones.sentiment_clone import SentimentClone
from mestre_dos_clones.clones.adaptive_clone import AdaptiveClone
from mestre_dos_clones.clones.meta_clone import MetaClone

# Import new clone types
from mestre_dos_clones.clones.regression_clone import RegressionClone
from mestre_dos_clones.clones.clustering_clone import ClusteringClone
from mestre_dos_clones.clones.dimension_clone import DimensionClone

# Import advanced utilities
from mestre_dos_clones.utils.evolutionary_islands import IslandManager
from mestre_dos_clones.utils.diversity_metrics import (
    calculate_clone_diversity,
    visualize_diversity_metrics,
    visualize_clone_diversity
)

def setup_output_dirs():
    """Creates the necessary directories for system output."""
    os.makedirs("output/clones", exist_ok=True)
    os.makedirs("output/results", exist_ok=True)
    os.makedirs("output/visualizations", exist_ok=True)
    os.makedirs("output/logs", exist_ok=True)
    os.makedirs("output/islands", exist_ok=True)


def save_population(population, generation, output_dir="output/clones"):
    """Saves the current population of clones."""
    generation_dir = os.path.join(output_dir, f"generation_{generation}")
    os.makedirs(generation_dir, exist_ok=True)
    
    for clone in population:
        filepath = os.path.join(generation_dir, f"{clone.name}.pkl")
        clone.save(filepath)
    
    return generation_dir


def save_evolution_history(history, output_path="output/results/evolution_history.json"):
    """Saves the evolution history to a JSON file."""
    # Prepare data for serialization
    serializable_history = []
    for gen_record in history:
        # Convert clones to IDs and basic information
        gen_data = {
            'generation': gen_record['generation'],
            'timestamp': gen_record['timestamp'],
            'population_size': len(gen_record['population']),
            'clones': [
                {
                    'id': clone.id,
                    'name': clone.name,
                    'score': float(clone.performance_score) if clone.performance_score is not None else None,
                    'algorithm': clone.algorithm_name,
                    'vectorizer': clone.vectorizer_name,
                    'generation': clone.generation,
                    'parent_ids': clone.parent_ids,
                    'type': clone.__class__.__name__
                }
                for clone in gen_record['population']
            ],
            'best_score': gen_record.get('best_score'),
            'avg_score': gen_record.get('avg_score'),
            'diversity_metrics': gen_record.get('diversity_metrics', {})
        }
        serializable_history.append(gen_data)
    
    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, indent=2, ensure_ascii=False)
    
    return output_path


def plot_evolution_progress(history, output_path="output/visualizations/evolution_progress.png"):
    """Generates a graph showing the evolution progress over generations."""
    generations = [rec['generation'] for rec in history]
    best_scores = [rec.get('best_score', 0) for rec in history]
    avg_scores = [rec.get('avg_score', 0) for rec in history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_scores, 'b-', label='Best Clone')
    plt.plot(generations, avg_scores, 'r-', label='Population Average')
    plt.xlabel('Generation')
    plt.ylabel('Score (F1)')
    plt.title('Evolution of Clone Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    
    return output_path


def run_evolution_cycle(
        num_generations=10,
        population_size=10,
        seed=42,
        use_islands=False,
        advanced_clones=True,
        enable_diversity_tracking=True,
        task_type="sentiment",  # New parameter: task type
        data_filepath=None,
        **task_params):  # Task-specific parameters
    """
    Executes the complete AGI evolution cycle.
    
    Args:
        num_generations: Number of generations to evolve
        population_size: Initial population size
        seed: Random seed for reproducibility
        use_islands: Whether to use the evolutionary islands system
        advanced_clones: Whether to allow advanced clones (AdaptiveClone, MetaClone)
        enable_diversity_tracking: Whether to track diversity metrics
        task_type: Task type ("sentiment", "prediction", "anomalies", etc.)
        data_filepath: Path to data file (optional)
        **task_params: Task-specific parameters
    """
    # Set reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Create output directories
    setup_output_dirs()
    
    # Initialize agents based on task type
    if task_type == "sentiment" or task_type == "classification":
        # Agents for sentiment classification (default task)
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type == "regression":
        # Agents for numerical regression
        print(f"Preparing agents for regression with metric {task_params.get('error_metric', 'MSE')}")
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use regression clones
        architect.available_clone_types = [RegressionClone]
        architect.clone_type_weights = [1.0]
    
    elif task_type == "clustering":
        # Agents for clustering
        print(f"Preparing agents for clustering with {task_params.get('n_clusters', 3)} clusters")
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use clustering clones
        architect.available_clone_types = [ClusteringClone]
        architect.clone_type_weights = [1.0]
    
    elif task_type == "dimension":
        # Agents for dimensionality reduction
        print(f"Preparing agents for dimensionality reduction with {task_params.get('n_components', 2)} components")
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use dimensionality reduction clones
        architect.available_clone_types = [DimensionClone]
        architect.clone_type_weights = [1.0]
    
    elif task_type == "prediction":
        # Agents for time series prediction (future implementation)
        print(f"Preparing agents for time series prediction with horizon of {task_params.get('forecast_horizon', 5)}")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type == "anomalies" and task_params.get("task_subtype") == "fraudes":
        # Agents for fraud detection (future implementation)
        print(f"Preparing agents for fraud detection with balance: {task_params.get('balance_classes', True)}")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type == "anomalies" and task_params.get("task_subtype") == "anomalies":
        # Agents for anomaly detection (future implementation)
        print(f"Preparing agents for anomaly detection with sensitivity: {task_params.get('sensitivity', 0.05)}")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type == "recommendation":
        # Agents for personalized recommendation (future implementation)
        print(f"Preparing agents for personalized recommendation")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type == "segmentation":
        # Agents for customer segmentation (future implementation)
        print(f"Preparing agents for segmentation with {task_params.get('num_clusters', 5)} groups")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    elif task_type in ["summarization", "response", "generation"]:
        # Agents for NLP tasks (future implementation)
        task_subtype = task_params.get("task_subtype", "")
        print(f"Preparing agents for language task: {task_type}/{task_subtype} with base model: {task_params.get('base_model', 'RNN')}")
        # Placeholder for future implementation
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    else:
        # If task type is not recognized, use default configuration for sentiment classification
        print(f"Task type '{task_type}' not recognized. Using default configuration for sentiment classification.")
        architect = Architect()
        trainer = Trainer()
        evaluator = Evaluator()
        
        # Configure the architect to use advanced clones
        if advanced_clones:
            architect.available_clone_types = [SentimentClone, AdaptiveClone, MetaClone]
            architect.clone_type_weights = [0.4, 0.4, 0.2]  # Weights for each type
    
    selector = Selector(
        elite_ratio=0.2,      # 20% of the best clones survive intact
        mutation_ratio=0.5,   # 50% are mutations
        crossover_ratio=0.2,  # 20% are crossovers
        random_ratio=0.1      # 10% are new random clones
    )
    
    # Initialize island manager if needed
    island_manager = None
    if use_islands:
        island_manager = IslandManager(
            migration_interval=2,  # Migrate every 2 generations
            migration_topology='ring'  # Ring topology
        )
    
    # Load data for training and evaluation
    print("Loading data...")
    
    # Load different types of data depending on the task
    if task_type == "sentiment" or task_type == "classification":
        # Load sentiment data (text + classification)
        texts, labels = trainer.load_data(data_filepath)
    
    elif task_type == "regression":
        # For regression, we use numerical data
        if data_filepath is None:
            # Create synthetic data for regression
            from sklearn.datasets import make_regression
            X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
        else:
            # Load data from file (CSV, for example)
            import pandas as pd
            df = pd.read_csv(data_filepath)
            
            # Assume the last column is the target
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values
    
    elif task_type == "clustering":
        # For clustering, we use numerical data without labels
        if data_filepath is None:
            # Create synthetic data for clustering
            from sklearn.datasets import make_blobs
            X, _ = make_blobs(n_samples=1000, centers=task_params.get('n_clusters', 3), random_state=42)
            y = np.zeros(X.shape[0])  # Fake labels (not used)
        else:
            # Load data from file (CSV, for example)
            import pandas as pd
            df = pd.read_csv(data_filepath)
            X = df.values
            y = np.zeros(X.shape[0])  # Fake labels (not used)
    
    elif task_type == "dimension":
        # For dimensionality reduction, we use high-dimensional data
        if data_filepath is None:
            # Create synthetic high-dimensional data
            from sklearn.datasets import make_classification
            X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, random_state=42)
        else:
            # Load data from file (CSV, for example)
            import pandas as pd
            df = pd.read_csv(data_filepath)
            X = df.values
            y = np.zeros(X.shape[0])  # Fake labels (may not be used)
    
    elif task_type == "prediction":
        # For time series prediction, we use the same mechanism temporarily
        # In a real implementation, this would be adapted to load time series
        print("Using example data for prediction. In a future implementation, this would load time series data.")
        texts, labels = trainer.load_data(data_filepath)
    
    elif task_type == "anomalies" and task_params.get("task_subtype") == "anomalies":
        # For anomaly detection, we use the same mechanism temporarily
        print("Using example data for anomaly detection. In a future implementation, this would load specific data.")
        texts, labels = trainer.load_data(data_filepath)
    
    elif task_type == "anomalies" and task_params.get("task_subtype") == "fraudes":
        # For fraud detection, we use the same mechanism temporarily
        print("Using example data for fraud detection. In a future implementation, this would load transaction data.")
        texts, labels = trainer.load_data(data_filepath)
    
    elif task_type in ["recommendation", "segmentation"]:
        # For recommendation/segmentation tasks, we use the same mechanism temporarily
        print(f"Using example data for {task_type}. In a future implementation, this would load appropriate data.")
        texts, labels = trainer.load_data(data_filepath)
    
    elif task_type in ["summarization", "response", "generation"]:
        # For NLP tasks, we use the same mechanism temporarily
        print(f"Using example data for {task_type} of {task_params.get('task_subtype', '')}. In a future implementation, this would load specific corpus.")
        texts, labels = trainer.load_data(data_filepath)
    
    else:
        # Load default data
        texts, labels = trainer.load_data(data_filepath)
    
    # Divide the data
    if task_type in ["regression", "clustering", "dimension"]:
        # For numeric tasks, we use a different split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.125, random_state=seed)
        
        data_splits = {
            'train': (X_train, y_train),
            'validation': (X_val, y_val),
            'test': (X_test, y_test)
        }
    else:
        # For text tasks, we use the existing split
        data_splits = trainer.prepare_data_splits(texts, labels, test_size=0.2, validation_size=0.1)
    
    # Define test data for the evaluator
    if task_type == 'regression':
        # For regression, we pass directly the X and y test vectors
        evaluator.set_test_data(data_splits['test'][0], data_splits['test'][1])
    elif task_type == 'clustering' or task_type == 'dimension':
        # For clustering and dimensionality reduction, we only pass the features
        # (the second argument can be None or an empty list, it will be ignored)
        evaluator.set_test_data(data_splits['test'][0], np.zeros(len(data_splits['test'][0])))
    else:
        # For text classification tasks, we use the traditional approach
        evaluator.set_test_data(*data_splits['test'])
    
    # Create initial population
    print("Creating initial population...")
    population = architect.create_initial_population(size=population_size)
    
    # Configure islands if using that system
    if use_islands:
        # Divide the population into 3 islands
        island_count = min(3, population_size // 3)
        island_size = population_size // island_count
        
        for i in range(island_count):
            start_idx = i * island_size
            end_idx = start_idx + island_size if i < island_count - 1 else len(population)
            island_population = population[start_idx:end_idx]
            
            island_manager.create_island(
                name=f"Island {i+1}",
                population=island_population,
                migration_rate=0.2  # 20% of the clones can migrate
            )
    
    # Evolution history
    evolution_history = []
    diversity_history = []
    
    # Execute the evolutionary cycle
    for generation in range(1, num_generations + 1):
        print(f"\n=== Generation {generation}/{num_generations} ===")
        start_time = time.time()
        
        # If using islands, evolve through them
        if use_islands:
            print(f"Evolving {len(island_manager.islands)} islands...")
            island_manager.evolve_all(
                selector, architect, trainer, evaluator, 
                data_splits['train'], target_size=island_size
            )
            # Get complete population from all islands
            population = island_manager.get_all_clones()
            
            # Show island information
            for i, island in enumerate(island_manager.islands):
                best_score = max([c.performance_score for c in island.population 
                                 if c.performance_score is not None], default=0)
                print(f"  Island {i+1}: {len(island.population)} clones, best score: {best_score:.4f}")
        else:
            # Normal flow without islands
            # Train the population
            print("Training clones...")
            trainer.train_population(population)
            
            # Evaluate the population
            print("Evaluating clones...")
            evaluation_results = evaluator.evaluate_population(population)
            
            # Classify clones by performance
            print("Classifying clones by performance...")
            ranked_clones = evaluator.rank_clones(population)
            
            # If not the last generation, evolve the population
            if generation < num_generations:
                print("Evolving to next generation...")
                # Evolve the population to the next generation
                population = selector.evolve_population(
                    ranked_clones,
                    architect,
                    target_size=population_size
                )
                
                # Train new clones
                for clone in population:
                    if not any(event['type'] == 'trained' for event in clone.history):
                        trainer.train_clone(clone, data_splits['train'][0], data_splits['train'][1])
                        evaluator.evaluate_clone(clone)
        
        # Evaluate entire population for consistent statistics
        if not use_islands:
            evaluation_results = evaluator.evaluate_population(population)
        ranked_clones = evaluator.rank_clones(population)
        
        # Calculate generation statistics
        best_clone, best_score = ranked_clones[0] if ranked_clones else (None, 0)
        avg_score = np.mean([score for _, score in ranked_clones]) if ranked_clones else 0
        
        print(f"Best clone: {best_clone.name if best_clone else 'N/A'} (Score: {best_score:.4f})")
        print(f"Average score: {avg_score:.4f}")
        
        # Calculate and record diversity metrics if enabled
        diversity_metrics = {}
        if enable_diversity_tracking:
            print("Calculating diversity metrics...")
            diversity_metrics = calculate_clone_diversity(population)
            diversity_history.append(diversity_metrics)
            
            overall_diversity = diversity_metrics['overall_diversity_index']
            algo_diversity = diversity_metrics['algorithm_diversity']['simpson_index']
            print(f"Overall diversity: {overall_diversity:.4f}")
            print(f"Algorithm diversity: {algo_diversity:.4f}")
            
            # Every 3 generations, generate diversity visualizations
            if generation % 3 == 0 or generation == num_generations:
                # Visualize diversity over time
                visualize_diversity_metrics(
                    diversity_history,
                    output_path=f"output/visualizations/diversity_trends_gen_{generation}.png",
                    show_plot=False
                )
                
                # Create clone diversity map
                if len(population) >= 5:  # Needs at least 5 clones to create visualization
                    visualize_clone_diversity(
                        population,
                        method='tsne',
                        output_path=f"output/visualizations/diversity_map_gen_{generation}.png",
                        show_plot=False
                    )
        
        # Record history for this generation
        generation_record = {
            'generation': generation,
            'timestamp': time.time(),
            'population': population.copy(),
            'best_clone_id': best_clone.id if best_clone else None,
            'best_score': best_score,
            'avg_score': avg_score,
            'evaluation_results': evaluation_results if not use_islands else None,
            'diversity_metrics': diversity_metrics
        }
        evolution_history.append(generation_record)
        
        # Save current population
        save_population(population, generation)
        
        # Save evaluation results
        if not use_islands and evaluation_results:
            evaluator.save_evaluation_results(
                evaluation_results,
                filepath=f"output/results/evaluation_gen_{generation}.json"
            )
        
        generation_time = time.time() - start_time
        print(f"Generation {generation} completed in {generation_time:.2f} seconds")
    
    # Save evolution history
    print("\nSaving evolution history...")
    save_evolution_history(evolution_history)
    
    # Generate progress graph
    print("Generating visualizations...")
    plot_evolution_progress(evolution_history)
    
    # If using island system, save its state
    if use_islands:
        island_manager.save_state()
    
    # Show final statistics
    best_clone_overall = None
    best_score_overall = float('-inf')
    
    for clone in population:
        if clone.performance_score is not None and clone.performance_score > best_score_overall:
            best_clone_overall = clone
            best_score_overall = clone.performance_score
    
    print("\nEvolution cycle completed!")
    print(f"Best clone: {best_clone_overall.name if best_clone_overall else 'N/A'} (Score: {best_score_overall:.4f})")
    if best_clone_overall:
        print(f"Type: {best_clone_overall.__class__.__name__}")
        print(f"Algorithm: {best_clone_overall.algorithm_name}")
    
    return evolution_history, best_clone_overall


def main():
    """Main function of the program."""
    parser = argparse.ArgumentParser(
        description="Clone Master - Self-evolving multi-agent AGI system"
    )
    parser.add_argument(
        "--generations", type=int, default=10,
        help="Number of generations to evolve (default: 10)"
    )
    parser.add_argument(
        "--population", type=int, default=10,
        help="Initial population size (default: 10)"
    )
    parser.add_argument(
        "--data", type=str, default=None,
        help="Path to data file (optional)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--islands", action="store_true",
        help="Use evolutionary islands system"
    )
    parser.add_argument(
        "--basic-only", action="store_true",
        help="Use only basic clones (SentimentClone)"
    )
    parser.add_argument(
        "--no-diversity", action="store_true",
        help="Disable diversity tracking"
    )
    
    args = parser.parse_args()
    
    print("==================================================")
    print("  🧬 Clone Master - AGI that Creates AGIs 🧬  ")
    print("==================================================")
    print(f"Generations: {args.generations}")
    print(f"Population size: {args.population}")
    print(f"Random seed: {args.seed}")
    print(f"Using island system: {'Yes' if args.islands else 'No'}")
    print(f"Clone types: {'Basic only' if args.basic_only else 'Advanced enabled'}")
    print(f"Diversity tracking: {'Disabled' if args.no_diversity else 'Enabled'}")
    print("==================================================")
    
    # Execute the evolutionary cycle
    run_evolution_cycle(
        num_generations=args.generations,
        population_size=args.population,
        data_filepath=args.data,
        seed=args.seed,
        use_islands=args.islands,
        advanced_clones=not args.basic_only,
        enable_diversity_tracking=not args.no_diversity
    )


if __name__ == "__main__":
    main()
