"""
Implementação de ilhas evolutivas para o sistema Mestre dos Clones.

Este módulo fornece mecanismos para criar e gerenciar múltiplas populações isoladas 
(ilhas) que evoluem separadamente, com migrações ocasionais de indivíduos entre elas.
"""

import random
import time
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set


class EvolutionaryIsland:
    """
    Representa uma ilha evolutiva com sua própria população de clones.
    """
    
    def __init__(self, 
                 name: str,
                 population: List,
                 migration_rate: float = 0.1,
                 island_id: Optional[str] = None):
        """
        Inicializa uma ilha evolutiva.
        
        Args:
            name: Nome da ilha
            population: Lista inicial de clones nesta ilha
            migration_rate: Taxa de migração (proporção de clones que podem migrar)
            island_id: ID opcional da ilha, gerado automaticamente se não fornecido
        """
        self.name = name
        self.id = island_id or f"island_{int(time.time())}_{random.randint(1000, 9999)}"
        self.population = population
        self.migration_rate = migration_rate
        self.generation = 0
        self.history = []
        
    def log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Registra um evento na história da ilha.
        
        Args:
            event_type: Tipo do evento ('immigration', 'emigration', 'evolution', etc)
            details: Detalhes do evento
            
        Returns:
            Dict: Registro do evento
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'island': self.id,
            'island_name': self.name,
            'generation': self.generation,
            'details': details or {}
        }
        self.history.append(event)
        return event
        
    def select_emigrants(self) -> List:
        """
        Seleciona clones desta ilha para migrar para outras ilhas.
        
        Returns:
            List: Clones selecionados para emigração
        """
        # Determina quantos clones vão emigrar (baseado na taxa de migração)
        emigrant_count = max(1, int(len(self.population) * self.migration_rate))
        
        # Escolhe aleatoriamente os emigrantes, com viés para os melhores
        # Primeiro ordena a população por desempenho (assume que cada clone tem performance_score)
        sorted_population = sorted(
            self.population, 
            key=lambda clone: clone.performance_score if clone.performance_score is not None else -1,
            reverse=True
        )
        
        # Probabilidade ponderada pela posição no ranking
        weights = np.linspace(1, 0.1, len(sorted_population))
        weights = weights / weights.sum()  # Normaliza
        
        # Seleciona emigrantes
        emigrant_indices = np.random.choice(
            range(len(sorted_population)),
            size=min(emigrant_count, len(sorted_population)),
            replace=False,
            p=weights
        )
        
        emigrants = [sorted_population[i] for i in emigrant_indices]
        
        # Remove os emigrantes da população
        self.population = [clone for clone in self.population if clone not in emigrants]
        
        # Registra o evento
        self.log_event('emigration', {
            'count': len(emigrants),
            'emigrant_ids': [clone.id for clone in emigrants]
        })
        
        return emigrants
        
    def receive_immigrants(self, immigrants: List) -> None:
        """
        Adiciona clones imigrantes à ilha.
        
        Args:
            immigrants: Lista de clones para adicionar à ilha
        """
        self.population.extend(immigrants)
        
        # Registra o evento
        self.log_event('immigration', {
            'count': len(immigrants),
            'immigrant_ids': [clone.id for clone in immigrants]
        })
    
    def evolve(self, 
               selector, 
               architect, 
               trainer,
               evaluator,
               data,
               target_size: Optional[int] = None) -> None:
        """
        Executa um ciclo de evolução nesta ilha.
        
        Args:
            selector: Agente Selecionador para escolher clones
            architect: Agente Arquiteto para criar novos clones
            trainer: Agente Treinador para treinar clones
            evaluator: Agente Avaliador para avaliar clones
            data: Dados para treinamento/avaliação
            target_size: Tamanho alvo da população após evolução
        """
        # Incrementa a geração
        self.generation += 1
        
        # Avalia clones que não têm pontuação (como os recém-imigrados)
        for clone in self.population:
            if clone.performance_score is None:
                # Primeiro verifica se o clone já foi treinado
                if not any(event['type'] == 'trained' for event in clone.history):
                    # Prepara os dados no formato esperado pelo trainer
                    training_data = {'train': (data[0], data[1])}
                    # Treina o clone antes de avaliar
                    trainer.train_clone(clone, data[0], data[1])
                # Avalia o clone após garantir que está treinado
                evaluator.evaluate_clone(clone)
        
        # Ordena a população atual por desempenho
        ranked_clones = [(clone, clone.performance_score) for clone in self.population]
        ranked_clones.sort(key=lambda x: x[1] if x[1] is not None else -1, reverse=True)
        
        # Evolui a população
        if target_size is None:
            target_size = len(self.population)
            
        # Usa o Selecionador para evoluir a população
        new_population = selector.evolve_population(
            ranked_clones, 
            architect,
            target_size
        )
        
        # Treina e avalia os novos clones
        for clone in new_population:
            # Pula clones já treinados (como os sobreviventes da geração anterior)
            if any(event['type'] == 'trained' for event in clone.history):
                continue
                
            # Treina e avalia novos clones
            training_data = {'train': (data[0], data[1])}
            trainer.train_clone(clone, data[0], data[1])
            evaluator.evaluate_clone(clone)
            
        # Atualiza a população da ilha
        self.population = new_population
        
        # Registra o evento
        self.log_event('evolution', {
            'population_size': len(self.population),
            'best_score': ranked_clones[0][1] if ranked_clones else None,
            'avg_score': np.mean([score for _, score in ranked_clones if score is not None])
        })


class IslandManager:
    """
    Gerencia múltiplas ilhas evolutivas e coordena migrações entre elas.
    """
    
    def __init__(self, 
                 islands: Optional[List[EvolutionaryIsland]] = None,
                 migration_interval: int = 3,
                 migration_topology: str = 'ring'):
        """
        Inicializa o gerenciador de ilhas.
        
        Args:
            islands: Lista inicial de ilhas
            migration_interval: A cada quantas gerações ocorre migração
            migration_topology: Topologia de migração ('ring', 'fully_connected', 'random')
        """
        self.islands = islands or []
        self.migration_interval = migration_interval
        self.migration_topology = migration_topology
        self.history = []
        
    def add_island(self, island: EvolutionaryIsland) -> None:
        """
        Adiciona uma nova ilha ao gerenciador.
        
        Args:
            island: Ilha a ser adicionada
        """
        self.islands.append(island)
        
    def create_island(self, 
                      name: str,
                      population: List,
                      migration_rate: float = 0.1) -> EvolutionaryIsland:
        """
        Cria e adiciona uma nova ilha.
        
        Args:
            name: Nome da ilha
            population: População inicial da ilha
            migration_rate: Taxa de migração
            
        Returns:
            EvolutionaryIsland: A ilha criada
        """
        island = EvolutionaryIsland(name, population, migration_rate)
        self.add_island(island)
        return island
    
    def log_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Registra um evento na história do gerenciador.
        
        Args:
            event_type: Tipo do evento
            details: Detalhes do evento
            
        Returns:
            Dict: Registro do evento
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details or {}
        }
        self.history.append(event)
        return event
        
    def should_migrate(self, generation: int) -> bool:
        """
        Determina se deve ocorrer migração nesta geração.
        
        Args:
            generation: Geração atual
            
        Returns:
            bool: True se deve migrar, False caso contrário
        """
        return generation % self.migration_interval == 0 and generation > 0
        
    def perform_migration(self) -> None:
        """
        Realiza a migração de clones entre ilhas de acordo com a topologia definida.
        """
        if len(self.islands) < 2:
            return  # Precisa de pelo menos duas ilhas para migração
            
        migration_map = {}  # Mapa de origem -> destino -> clones
        
        # Coleta emigrantes de cada ilha
        all_emigrants = {island.id: island.select_emigrants() for island in self.islands}
        
        # Determina destinos de acordo com a topologia
        if self.migration_topology == 'ring':
            # Cada ilha envia para a próxima no anel
            for i, source_island in enumerate(self.islands):
                dest_island = self.islands[(i + 1) % len(self.islands)]
                migration_map[source_island.id] = {dest_island.id: all_emigrants[source_island.id]}
                
        elif self.migration_topology == 'fully_connected':
            # Cada ilha envia para todas as outras
            for source_island in self.islands:
                migration_map[source_island.id] = {}
                emigrants = all_emigrants[source_island.id]
                
                # Distribui emigrantes entre todas as outras ilhas
                if emigrants:
                    dest_islands = [island for island in self.islands if island.id != source_island.id]
                    emigrants_per_island = max(1, len(emigrants) // len(dest_islands))
                    
                    for i, dest_island in enumerate(dest_islands):
                        start_idx = i * emigrants_per_island
                        end_idx = (i + 1) * emigrants_per_island if i < len(dest_islands) - 1 else len(emigrants)
                        migration_map[source_island.id][dest_island.id] = emigrants[start_idx:end_idx]
                        
        elif self.migration_topology == 'random':
            # Cada ilha envia aleatoriamente para outras
            for source_island in self.islands:
                migration_map[source_island.id] = {}
                emigrants = all_emigrants[source_island.id]
                
                if emigrants:
                    # Para cada emigrante, escolhe uma ilha destino aleatória
                    for emigrant in emigrants:
                        dest_island = random.choice([island for island in self.islands if island.id != source_island.id])
                        
                        if dest_island.id not in migration_map[source_island.id]:
                            migration_map[source_island.id][dest_island.id] = []
                            
                        migration_map[source_island.id][dest_island.id].append(emigrant)
        
        # Realiza a migração
        migration_count = 0
        for source_id, destinations in migration_map.items():
            for dest_id, emigrants in destinations.items():
                if emigrants:
                    dest_island = next(island for island in self.islands if island.id == dest_id)
                    dest_island.receive_immigrants(emigrants)
                    migration_count += len(emigrants)
        
        # Registra o evento
        self.log_event('migration', {
            'topology': self.migration_topology,
            'total_migrations': migration_count,
            'migration_map': {
                source: {dest: len(emigrants) for dest, emigrants in destinations.items()}
                for source, destinations in migration_map.items()
            }
        })
    
    def evolve_all(self, 
                  selector, 
                  architect, 
                  trainer,
                  evaluator,
                  data,
                  target_size: Optional[int] = None) -> None:
        """
        Evolui todas as ilhas e realiza migração se necessário.
        
        Args:
            selector: Agente Selecionador
            architect: Agente Arquiteto
            trainer: Agente Treinador
            evaluator: Agente Avaliador
            data: Dados para treinamento/avaliação
            target_size: Tamanho alvo da população após evolução
        """
        # Determina a geração atual (máxima entre todas as ilhas)
        current_generation = max(island.generation for island in self.islands) if self.islands else 0
        
        # Verifica se deve ocorrer migração
        if self.should_migrate(current_generation):
            self.perform_migration()
            
        # Evolui cada ilha
        for island in self.islands:
            island.evolve(selector, architect, trainer, evaluator, data, target_size)
            
        # Registra o evento
        self.log_event('evolution_cycle', {
            'generation': current_generation + 1,
            'islands': len(self.islands),
            'island_populations': {island.name: len(island.population) for island in self.islands},
            'island_best_scores': {
                island.name: max(
                    (clone.performance_score for clone in island.population if clone.performance_score is not None),
                    default=None
                )
                for island in self.islands
            }
        })
    
    def get_best_clone(self) -> Tuple:
        """
        Retorna o melhor clone entre todas as ilhas.
        
        Returns:
            Tuple: (ilha, clone, pontuação)
        """
        best_clone = None
        best_score = float('-inf')
        best_island = None
        
        for island in self.islands:
            for clone in island.population:
                if clone.performance_score is not None and clone.performance_score > best_score:
                    best_clone = clone
                    best_score = clone.performance_score
                    best_island = island
                    
        return best_island, best_clone, best_score
    
    def get_all_clones(self) -> List:
        """
        Retorna todos os clones de todas as ilhas.
        
        Returns:
            List: Lista de todos os clones
        """
        all_clones = []
        for island in self.islands:
            all_clones.extend(island.population)
        return all_clones
    
    def save_state(self, directory: str = "output/islands") -> str:
        """
        Salva o estado do gerenciador de ilhas.
        
        Args:
            directory: Diretório onde salvar
            
        Returns:
            str: Caminho do arquivo onde o estado foi salvo
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Salva os metadados (não inclui os objetos clones)
        metadata = {
            'timestamp': time.time(),
            'migration_interval': self.migration_interval,
            'migration_topology': self.migration_topology,
            'islands': [
                {
                    'id': island.id,
                    'name': island.name,
                    'generation': island.generation,
                    'population_size': len(island.population),
                    'migration_rate': island.migration_rate
                }
                for island in self.islands
            ],
            'history': self.history
        }
        
        metadata_path = os.path.join(directory, f"island_manager_{int(time.time())}.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Salva cada ilha e seus clones
        for island in self.islands:
            island_dir = os.path.join(directory, island.id)
            if not os.path.exists(island_dir):
                os.makedirs(island_dir)
                
            # Salva metadados da ilha
            island_meta = {
                'id': island.id,
                'name': island.name,
                'generation': island.generation,
                'migration_rate': island.migration_rate,
                'history': island.history,
                'clones': [clone.id for clone in island.population]
            }
            
            with open(os.path.join(island_dir, "metadata.json"), 'w') as f:
                json.dump(island_meta, f, indent=2)
                
            # Salva cada clone
            for clone in island.population:
                clone.save(os.path.join(island_dir, "clones"))
                
        return metadata_path 
