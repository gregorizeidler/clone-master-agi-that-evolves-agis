"""
Utilitários para o sistema Mestre dos Clones.

Este módulo contém funções auxiliares para visualização, análise e 
manipulação de dados relacionados aos clones e seu processo evolutivo.
Inclui também implementações de algoritmos evolutivos avançados como o sistema de ilhas.
"""

from mestre_dos_clones.utils.visualization import (
    plot_evolution_progress,
    plot_generation_metrics,
    create_genealogy_graph
)

from mestre_dos_clones.utils.evolutionary_islands import (
    EvolutionaryIsland,
    IslandManager
)

from mestre_dos_clones.utils.diversity_metrics import (
    calculate_clone_diversity,
    calculate_algorithm_diversity,
    calculate_hyperparameter_diversity,
    calculate_performance_diversity,
    visualize_algorithm_distribution,
    visualize_diversity_metrics,
    visualize_clone_diversity
)

__all__ = [
    # Visualização básica
    'plot_evolution_progress',
    'plot_generation_metrics',
    'create_genealogy_graph',
    
    # Ilhas evolutivas
    'EvolutionaryIsland',
    'IslandManager',
    
    # Métricas de diversidade
    'calculate_clone_diversity',
    'calculate_algorithm_diversity',
    'calculate_hyperparameter_diversity',
    'calculate_performance_diversity',
    'visualize_algorithm_distribution',
    'visualize_diversity_metrics',
    'visualize_clone_diversity'
]
