"""
Módulo contendo as implementações de clones (AGIs especializadas).

Este módulo define:
- BaseClone: Classe abstrata que define a interface comum a todos os clones.
- SentimentClone: Implementação específica para classificação de sentimentos.
- AdaptiveClone: Clone com capacidade de auto-adaptação de parâmetros evolutivos.
- MetaClone: Clone com capacidades de meta-aprendizado para adaptar sua estratégia.
"""

from mestre_dos_clones.clones.base_clone import BaseClone
from mestre_dos_clones.clones.sentiment_clone import SentimentClone
from mestre_dos_clones.clones.adaptive_clone import AdaptiveClone
from mestre_dos_clones.clones.meta_clone import MetaClone

__all__ = ['BaseClone', 'SentimentClone', 'AdaptiveClone', 'MetaClone']
