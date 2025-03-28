#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Command line interface to interact with the Clone Master system.

Allows analyzing results, visualizing evolution, testing individual clones and more.
"""

import os
import argparse
import pickle
import json
import time
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import glob

# Import the necessary modules
from mestre_dos_clones.clones.sentiment_clone import SentimentClone
from mestre_dos_clones.clones.adaptive_clone import AdaptiveClone
from mestre_dos_clones.clones.meta_clone import MetaClone
from mestre_dos_clones.clones.base_clone import BaseClone
from mestre_dos_clones.utils.diversity_metrics import (
    calculate_clone_diversity,
    visualize_algorithm_distribution,
    visualize_diversity_metrics,
    visualize_clone_diversity
)
from mestre_dos_clones.utils.evolutionary_islands import IslandManager

OUTPUT_DIR = "output"
GENERATIONS_DIR = os.path.join(OUTPUT_DIR, "generations")
VISUALIZATIONS_DIR = os.path.join(OUTPUT_DIR, "visualizations")
ISLANDS_DIR = os.path.join(OUTPUT_DIR, "islands")

# Function to list all available generations
def list_generations() -> List[str]:
    """Lists all available generations."""
    if not os.path.exists(GENERATIONS_DIR):
        return []
    
    generations = []
    for dir_path in glob.glob(os.path.join(GENERATIONS_DIR, "*")):
        if os.path.isdir(dir_path):
            generations.append(os.path.basename(dir_path))
    
    return sorted(generations)

# Function to list all clones in a generation
def list_clones(generation: str) -> List[str]:
    """Lists all clones from a specific generation."""
    gen_dir = os.path.join(GENERATIONS_DIR, generation)
    if not os.path.exists(gen_dir):
        return []
    
    clones = []
    for file_path in glob.glob(os.path.join(gen_dir, "*.pkl")):
        clones.append(os.path.basename(file_path).replace(".pkl", ""))
    
    return sorted(clones)

# Function to list all available islands
def list_islands() -> List[str]:
    """Lists all available islands."""
    if not os.path.exists(ISLANDS_DIR):
        return []
    
    islands = []
    for file_path in glob.glob(os.path.join(ISLANDS_DIR, "*.json")):
        islands.append(os.path.basename(file_path).replace(".json", ""))
    
    return sorted(islands)

# Function to load a specific clone
def load_clone(generation: str, clone_name: str) -> Optional[BaseClone]:
    """Loads a specific clone from the indicated generation."""
    clone_path = os.path.join(GENERATIONS_DIR, generation, f"{clone_name}.pkl")
    if not os.path.exists(clone_path):
        return None
    
    try:
        with open(clone_path, 'rb') as f:
            clone = pickle.load(f)
        return clone
    except Exception as e:
        print(f"Error loading clone: {e}")
        return None

# Function to test a clone
def test_clone(clone: BaseClone, text: str) -> Dict[str, Any]:
    """Tests a clone with the provided text and returns the prediction."""
    if not clone.is_trained:
        return {
            "success": False,
            "error": "Clone not trained",
            "sentiment": None,
            "confidence": None
        }
    
    try:
        sentiment, confidence = clone.predict(text)
        
        return {
            "success": True,
            "sentiment": "positive" if sentiment == 1 else "negative",
            "confidence": confidence,
            "clone_info": {
                "name": clone.name,
                "id": clone.id,
                "algorithm": clone.algorithm_name,
                "vectorizer": clone.vectorizer_name,
                "type": clone.__class__.__name__
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "sentiment": None,
            "confidence": None
        }

# Command to visualize evolution metrics
def command_visualize(args):
    """Generates visualizations of clone evolution."""
    history_path = os.path.join(OUTPUT_DIR, "evolution_history.json")
    
    if not os.path.exists(history_path):
        print("Evolution history file not found.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    # Visualize evolution progress
    plt.figure(figsize=(10, 6))
    
    generations = [entry['generation'] for entry in history]
    best_scores = [entry['best_score'] for entry in history]
    avg_scores = [entry['avg_score'] for entry in history]
    
    plt.plot(generations, best_scores, 'b-', label='Best score')
    plt.plot(generations, avg_scores, 'r-', label='Average score')
    
    plt.xlabel('Generation')
    plt.ylabel('Score')
    plt.title('Clone Evolution Progress')
    plt.legend()
    plt.grid(True)
    
    vis_path = os.path.join(VISUALIZATIONS_DIR, "evolution.png")
    plt.savefig(vis_path)
    plt.close()
    
    print(f"Visualization saved at {vis_path}")
    
    # If requested to include diversity visualizations
    if args.diversity:
        # Load clones from the last generation
        last_gen = max(list_generations(), key=lambda x: int(x.split("_")[1]))
        clone_names = list_clones(last_gen)
        clones = [load_clone(last_gen, name) for name in clone_names]
        clones = [c for c in clones if c is not None]
        
        if clones:
            # Visualize algorithm distribution
            alg_vis_path = os.path.join(VISUALIZATIONS_DIR, "algorithms.png")
            visualize_algorithm_distribution(clones, alg_vis_path)
            print(f"Algorithm visualization saved at {alg_vis_path}")
            
            # Visualize clone diversity using t-SNE
            div_vis_path = os.path.join(VISUALIZATIONS_DIR, "diversity_tsne.png")
            visualize_clone_diversity(clones, method='tsne', output_path=div_vis_path)
            print(f"Diversity visualization (t-SNE) saved at {div_vis_path}")
            
            # Visualize clone diversity using PCA
            div_pca_path = os.path.join(VISUALIZATIONS_DIR, "diversity_pca.png")
            visualize_clone_diversity(clones, method='pca', output_path=div_pca_path)
            print(f"Diversity visualization (PCA) saved at {div_pca_path}")
            
            # Calculate and display diversity metrics
            diversity_metrics = calculate_clone_diversity(clones)
            print("\nPopulation Diversity Metrics:")
            for metric, value in diversity_metrics.items():
                print(f"- {metric}: {value:.4f}")

# Command to list generations or clones
def command_list(args):
    """Lists generations or clones based on user input."""
    if args.type == 'generations':
        generations = list_generations()
        if generations:
            print("Available generations:")
            for gen in generations:
                print(f"- {gen}")
        else:
            print("No generations found.")
    
    elif args.type == 'clones':
        if not args.generation:
            print("Error: To list clones, specify the generation with --generation")
            return
        
        clones = list_clones(args.generation)
        if clones:
            print(f"Clones from generation {args.generation}:")
            for clone in clones:
                print(f"- {clone}")
        else:
            print(f"No clones found in generation {args.generation}.")
    
    elif args.type == 'islands':
        islands = list_islands()
        if islands:
            print("Available evolutionary islands:")
            for island in islands:
                print(f"- {island}")
        else:
            print("No evolutionary islands found.")

# Command to test a clone
def command_test(args):
    """Tests a specific clone interactively or with provided text."""
    if not args.generation or not args.clone:
        print("Error: Specify the generation and clone name to test.")
        return
    
    clone = load_clone(args.generation, args.clone)
    if not clone:
        print(f"Error: Clone {args.clone} not found in generation {args.generation}.")
        return
    
    # Show clone information
    print(f"\nClone Information:")
    print(f"- Name: {clone.name}")
    print(f"- ID: {clone.id}")
    print(f"- Type: {clone.__class__.__name__}")
    print(f"- Algorithm: {clone.algorithm_name}")
    print(f"- Vectorizer: {clone.vectorizer_name}")
    
    # Show specific information based on clone type
    if isinstance(clone, AdaptiveClone):
        print("\nMutation Rates:")
        for param, value in clone.mutation_rates.items():
            print(f"- {param}: {value:.4f}")
    
    if isinstance(clone, MetaClone):
        print("\nMeta-Learning Strategies:")
        for strategy, config in clone.meta_strategies.items():
            print(f"- {strategy}: {config}")
    
    # Test interactively or with provided text
    if args.text:
        result = test_clone(clone, args.text)
        
        print("\nPrediction:")
        if result["success"]:
            print(f"- Sentiment: {result['sentiment']}")
            print(f"- Confidence: {result['confidence']:.4f}")
        else:
            print(f"- Error: {result['error']}")
    else:
        print("\nInteractive test mode. Enter 'exit' to quit.")
        
        while True:
            text = input("\nEnter text to analyze: ")
            if text.lower() == 'exit':
                break
            
            result = test_clone(clone, text)
            
            print("\nPrediction:")
            if result["success"]:
                print(f"- Sentiment: {result['sentiment']}")
                print(f"- Confidence: {result['confidence']:.4f}")
            else:
                print(f"- Error: {result['error']}")

# Command to show best clone info
def command_best(args):
    """Shows information about the best performing clone."""
    history_path = os.path.join(OUTPUT_DIR, "evolution_history.json")
    
    if not os.path.exists(history_path):
        print("Evolution history file not found.")
        return
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    if not history:
        print("No evolution data found.")
        return
    
    # Find the entry with the highest score
    best_entry = max(history, key=lambda x: x.get('best_score', 0))
    
    print("\nBest Clone Information:")
    print(f"- Generation: {best_entry.get('generation', 'Unknown')}")
    print(f"- Score: {best_entry.get('best_score', 0):.4f}")
    
    if 'best_clone' in best_entry:
        best_clone_info = best_entry['best_clone']
        print(f"- Name: {best_clone_info.get('name', 'Unknown')}")
        print(f"- ID: {best_clone_info.get('id', 'Unknown')}")
        print(f"- Type: {best_clone_info.get('type', 'Unknown')}")
        print(f"- Algorithm: {best_clone_info.get('algorithm', 'Unknown')}")

# Command to show island information
def command_island_info(args):
    """Shows information about the evolutionary islands."""
    if not args.island:
        print("Error: Specify the island name with --island")
        
        # Show available islands
        islands = list_islands()
        if islands:
            print("\nAvailable islands:")
            for island in islands:
                print(f"- {island}")
        return
    
    island_path = os.path.join(ISLANDS_DIR, f"{args.island}.json")
    if not os.path.exists(island_path):
        print(f"Error: Island {args.island} not found.")
        return
    
    try:
        with open(island_path, 'r') as f:
            island_data = json.load(f)
        
        print(f"\nIsland Information: {args.island}")
        print(f"- Population Size: {len(island_data.get('population', []))}")
        print(f"- Generation: {island_data.get('generation', 'Unknown')}")
        
        if 'best_clone' in island_data:
            best = island_data['best_clone']
            print("\nBest Clone:")
            print(f"- Name: {best.get('name', 'Unknown')}")
            print(f"- ID: {best.get('id', 'Unknown')}")
            print(f"- Score: {best.get('score', 0):.4f}")
            print(f"- Type: {best.get('type', 'Unknown')}")
            print(f"- Algorithm: {best.get('algorithm', 'Unknown')}")
        
        if 'migration_history' in island_data:
            migrations = island_data['migration_history']
            print(f"\nMigrations: {len(migrations)}")
            
            for i, migration in enumerate(migrations[-5:], 1):  # Show last 5 migrations
                print(f"\nMigration {i}:")
                print(f"- From: Island {migration.get('from_island', 'Unknown')}")
                print(f"- To: Island {migration.get('to_island', 'Unknown')}")
                print(f"- Clone: {migration.get('clone_name', 'Unknown')}")
                print(f"- Generation: {migration.get('generation', 'Unknown')}")
        
        if 'diversity_metrics' in island_data:
            diversity = island_data['diversity_metrics']
            print("\nDiversity Metrics:")
            for metric, value in diversity.items():
                print(f"- {metric}: {value:.4f}")
        
    except Exception as e:
        print(f"Error loading island data: {e}")

def main():
    """Main function to parse arguments and execute commands."""
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Command line interface for the Clone Master system."
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List generations, clones, or islands')
    list_parser.add_argument('type', choices=['generations', 'clones', 'islands'], 
                          help='The type of items to list')
    list_parser.add_argument('--generation', '-g', help='Generation to list clones from')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a specific clone')
    test_parser.add_argument('--generation', '-g', required=True, help='Generation of the clone')
    test_parser.add_argument('--clone', '-c', required=True, help='Name of the clone to test')
    test_parser.add_argument('--text', '-t', help='Text to analyze (if not provided, enters interactive mode)')
    
    # Visualize command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations of the evolution')
    visualize_parser.add_argument('--diversity', '-d', action='store_true', 
                               help='Include diversity visualizations')
    
    # Best command
    best_parser = subparsers.add_parser('best', help='Show information about the best performing clone')
    
    # Island info command
    island_parser = subparsers.add_parser('island', help='Show information about the evolutionary islands')
    island_parser.add_argument('--island', '-i', help='Name of the island to show information about')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure output directories exist
    os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
    
    # Execute the appropriate command
    if args.command == 'list':
        command_list(args)
    elif args.command == 'test':
        command_test(args)
    elif args.command == 'visualize':
        command_visualize(args)
    elif args.command == 'best':
        command_best(args)
    elif args.command == 'island':
        command_island_info(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
