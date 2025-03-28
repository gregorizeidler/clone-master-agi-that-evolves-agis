import random
from typing import Dict, List, Any, Optional, Tuple, Set
import numpy as np
import time

from mestre_dos_clones.clones.base_clone import BaseClone
from mestre_dos_clones.clones.sentiment_clone import SentimentClone


class Selector:
    """
    Selector Agent responsible for choosing which clones survive,
    are modified, or discarded based on their performance.
    Implements the evolutionary logic of the system.
    """
    
    def __init__(self, 
                 elite_ratio: float = 0.2,
                 mutation_ratio: float = 0.4,
                 crossover_ratio: float = 0.2,
                 random_ratio: float = 0.1):
        """
        Initializes the Selector.
        
        Args:
            elite_ratio: Proportion of elite clones that survive intact
            mutation_ratio: Proportion of clones that will undergo mutation
            crossover_ratio: Proportion of clones that will undergo crossover
            random_ratio: Proportion of new random clones
        """
        self.elite_ratio = elite_ratio
        self.mutation_ratio = mutation_ratio
        self.crossover_ratio = crossover_ratio
        self.random_ratio = random_ratio
        self.history = []
        
    def log_action(self, action_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Records an action by the Selector.
        
        Args:
            action_type: Type of action ('select', 'evolve', etc)
            details: Action details
            
        Returns:
            Dict: Action record
        """
        log_entry = {
            'agent': 'selector',
            'action': action_type,
            'timestamp': time.time(),
            'details': details or {}
        }
        self.history.append(log_entry)
        return log_entry
    
    def select_elite(self, ranked_clones: List[Tuple[SentimentClone, float]], 
                     count: int) -> List[SentimentClone]:
        """
        Selects the best clones to survive intact.
        
        Args:
            ranked_clones: List of (clone, score) ordered by performance
            count: Number of clones to select
            
        Returns:
            List[SentimentClone]: List of elite clones
        """
        # Select the 'count' best clones
        elite = [clone for clone, _ in ranked_clones[:count]]
        
        # Record the selection
        self.log_action('select_elite', {
            'count': count,
            'selected_ids': [clone.id for clone in elite]
        })
        
        return elite
    
    def select_for_mutation(self, ranked_clones: List[Tuple[SentimentClone, float]], 
                           count: int,
                           elite_ids: Set[str] = None) -> List[SentimentClone]:
        """
        Selects clones for mutation, with bias towards the best ones.
        
        Args:
            ranked_clones: List of (clone, score) ordered by performance
            count: Number of clones to select
            elite_ids: IDs of elite clones (to avoid duplication)
            
        Returns:
            List[SentimentClone]: List of clones for mutation
        """
        elite_ids = elite_ids or set()
        
        # Create list of all eligible clones (non-elite)
        eligible_clones = [clone for clone, _ in ranked_clones 
                          if clone.id not in elite_ids]
        
        if not eligible_clones:
            return []
            
        # Calculate probabilities with bias towards better clones
        # The better the clone, the higher the chance of being selected
        probs = np.linspace(1, 0.1, len(eligible_clones))
        probs = probs / probs.sum()  # Normalize to sum 1
        
        # Select clones with probability weighting
        # Allows the same clone to be selected multiple times
        selected_indices = np.random.choice(
            range(len(eligible_clones)), 
            size=min(count, len(eligible_clones)),
            replace=False,  # Without replacement
            p=probs
        )
        
        selected = [eligible_clones[i] for i in selected_indices]
        
        # Record the selection
        self.log_action('select_for_mutation', {
            'count': count,
            'selected_ids': [clone.id for clone in selected]
        })
        
        return selected
    
    def select_for_crossover(self, ranked_clones: List[Tuple[SentimentClone, float]], 
                            count: int,
                            excluded_ids: Set[str] = None) -> List[Tuple[SentimentClone, SentimentClone]]:
        """
        Selects pairs of clones for crossover.
        
        Args:
            ranked_clones: List of (clone, score) ordered by performance
            count: Number of pairs to select
            excluded_ids: IDs of clones already selected for other purposes
            
        Returns:
            List[Tuple[SentimentClone, SentimentClone]]: List of pairs for crossover
        """
        excluded_ids = excluded_ids or set()
        
        # Create list of all eligible clones
        eligible_clones = [clone for clone, _ in ranked_clones 
                          if clone.id not in excluded_ids]
        
        if len(eligible_clones) < 2:
            return []
            
        # Calculate probabilities with bias towards better clones
        probs = np.linspace(1, 0.1, len(eligible_clones))
        probs = probs / probs.sum()  # Normalize to sum 1
        
        # Select pairs for crossover
        pairs = []
        for _ in range(min(count, len(eligible_clones) // 2)):
            # Select two distinct clones with weighted probability
            indices = np.random.choice(
                range(len(eligible_clones)), 
                size=2, 
                replace=False,
                p=probs
            )
            pairs.append((eligible_clones[indices[0]], eligible_clones[indices[1]]))
        
        # Record the selection
        self.log_action('select_for_crossover', {
            'count': len(pairs),
            'selected_pairs': [
                (clone1.id, clone2.id) for clone1, clone2 in pairs
            ]
        })
        
        return pairs
    
    def crossover(self, parent1: SentimentClone, parent2: SentimentClone) -> SentimentClone:
        """
        Performs crossover between two clones to generate a new one.
        
        Args:
            parent1: First parent clone
            parent2: Second parent clone
            
        Returns:
            SentimentClone: Child clone resulting from crossover
        """
        # Randomly decide the algorithm and vectorizer
        if random.random() < 0.5:
            new_algorithm = parent1.algorithm_name
        else:
            new_algorithm = parent2.algorithm_name
            
        if random.random() < 0.5:
            new_vectorizer = parent1.vectorizer_name
        else:
            new_vectorizer = parent2.vectorizer_name
        
        # Mix hyperparameters from parents
        new_hyperparams = {}
        
        # Determine which hyperparameters exist in both parents
        all_keys = set(parent1.hyperparams.keys()) | set(parent2.hyperparams.keys())
        
        for key in all_keys:
            # If both parents have the parameter, choose one or average them
            if key in parent1.hyperparams and key in parent2.hyperparams:
                # For numeric parameters, we can average or choose one
                if isinstance(parent1.hyperparams[key], (int, float)) and isinstance(parent2.hyperparams[key], (int, float)):
                    # 50% chance of averaging, 50% chance of choosing one parent
                    if random.random() < 0.5:
                        # Random weighted average
                        weight = random.random()  # Between 0 and 1
                        value = weight * parent1.hyperparams[key] + (1 - weight) * parent2.hyperparams[key]
                        # Convert to int if either parent has an int value
                        if isinstance(parent1.hyperparams[key], int) or isinstance(parent2.hyperparams[key], int):
                            value = int(round(value))
                    else:
                        # Choose one of the parents
                        value = parent1.hyperparams[key] if random.random() < 0.5 else parent2.hyperparams[key]
                else:
                    # For non-numeric parameters, choose one of the parents
                    value = parent1.hyperparams[key] if random.random() < 0.5 else parent2.hyperparams[key]
            # If only one parent has the parameter, use that value
            elif key in parent1.hyperparams:
                value = parent1.hyperparams[key]
            else:
                value = parent2.hyperparams[key]
                
            new_hyperparams[key] = value
        
        # Create the child clone
        child = SentimentClone(
            algorithm=new_algorithm,
            vectorizer=new_vectorizer,
            hyperparams=new_hyperparams,
            name=f"Crossover-{parent1.id}-{parent2.id}-{str(random.randint(1000, 9999))}"
        )
        
        # Configure parent relationship
        child.parent_ids = [parent1.id, parent2.id]
        child.generation = max(parent1.generation, parent2.generation) + 1
        
        # Record the crossover
        child.log_event('crossover', {
            'parent1_id': parent1.id,
            'parent2_id': parent2.id,
            'algorithm': new_algorithm,
            'vectorizer': new_vectorizer,
            'hyperparams': new_hyperparams
        })
        
        # Record the action
        self.log_action('crossover', {
            'parent1_id': parent1.id,
            'parent2_id': parent2.id,
            'child_id': child.id
        })
        
        return child
    
    def evolve_population(self, ranked_clones: List[Tuple[SentimentClone, float]], 
                         architect,
                         target_size: int = None) -> List[SentimentClone]:
        """
        Evolves a population of clones, applying natural selection and variation.
        
        Args:
            ranked_clones: List of (clone, score) ordered by performance
            architect: Architect Agent to create/modify clones
            target_size: Target size of the new population
            
        Returns:
            List[SentimentClone]: New generation of clones
        """
        if not ranked_clones:
            return []
            
        current_size = len(ranked_clones)
        target_size = target_size or current_size
        
        # Calculate how many clones of each type we'll have
        elite_count = max(1, int(target_size * self.elite_ratio))
        mutation_count = max(1, int(target_size * self.mutation_ratio))
        crossover_count = max(0, int(target_size * self.crossover_ratio))
        random_count = max(0, int(target_size * self.random_ratio))
        
        # Adjust to ensure we have exactly target_size clones
        remaining = target_size - (elite_count + mutation_count + crossover_count + random_count)
        if remaining > 0:
            # Distribute the extra clones, prioritizing elites and mutations
            if elite_count < current_size * 0.5:
                elite_count += remaining
            else:
                mutation_count += remaining
        elif remaining < 0:
            # Remove extra clones, prioritizing keeping elites
            if random_count >= abs(remaining):
                random_count += remaining  # (remaining is negative)
            elif crossover_count >= abs(remaining):
                crossover_count += remaining
            elif mutation_count > abs(remaining):
                mutation_count += remaining
            else:
                elite_count += remaining
                
        # 1. Select elites (survive intact)
        elites = self.select_elite(ranked_clones, elite_count)
        elite_ids = {clone.id for clone in elites}
        
        # 2. Select clones for mutation
        mutation_candidates = self.select_for_mutation(ranked_clones, mutation_count, elite_ids)
        
        # 3. Generate mutants
        mutants = []
        for clone in mutation_candidates:
            mutated = architect.modify_clone(clone, modification_type='mutate')
            mutants.append(mutated)
            
        # 4. Select pairs for crossover
        excluded_ids = elite_ids | {clone.id for clone in mutation_candidates}
        crossover_pairs = self.select_for_crossover(ranked_clones, crossover_count, excluded_ids)
        
        # 5. Generate offspring by crossover
        crossover_children = []
        for parent1, parent2 in crossover_pairs:
            child = self.crossover(parent1, parent2)
            crossover_children.append(child)
            
        # 6. Generate random clones
        random_clones = []
        for _ in range(random_count):
            random_clone = architect.create_sentiment_clone('random')
            random_clones.append(random_clone)
            
        # Combine all clones to form the new population
        new_population = elites + mutants + crossover_children + random_clones
        
        # Record the evolution
        self.log_action('evolve_population', {
            'old_size': current_size,
            'new_size': len(new_population),
            'elite_count': len(elites),
            'mutant_count': len(mutants),
            'crossover_count': len(crossover_children),
            'random_count': len(random_clones)
        })
        
        return new_population 
