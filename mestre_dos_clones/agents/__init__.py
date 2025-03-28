"""
Agentes responsáveis pela criação, treinamento, avaliação e evolução dos clones.

Este módulo contém os quatro agentes principais do sistema:
- Architect: Projeta novos clones, definindo sua estrutura e lógica.
- Trainer: Treina os clones com dados para executar tarefas específicas.
- Evaluator: Testa clones em cenários controlados e mede seu desempenho.
- Selector: Decide quais clones sobrevivem, evoluem ou são descartados.
"""

from mestre_dos_clones.agents.architect import Architect
from mestre_dos_clones.agents.trainer import Trainer
from mestre_dos_clones.agents.evaluator import Evaluator
from mestre_dos_clones.agents.selector import Selector

__all__ = ['Architect', 'Trainer', 'Evaluator', 'Selector']
