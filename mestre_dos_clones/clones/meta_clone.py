"""
Clone com capacidades de meta-aprendizado para o sistema Mestre dos Clones.

Este módulo implementa um clone que pode ajustar sua própria estratégia
de aprendizado com base na experiência anterior.
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import random
import json
from typing import Dict, List, Any, Optional

from mestre_dos_clones.clones.adaptive_clone import AdaptiveClone


class MetaClone(AdaptiveClone):
    """
    Clone com capacidade de meta-aprendizado que pode ajustar sua estratégia
    de aprendizado com base no desempenho em diferentes cenários.
    """
    
    def __init__(self, 
                 algorithm='naive_bayes', 
                 vectorizer='count',
                 hyperparams=None,
                 mutation_rates=None,
                 meta_strategies=None,
                 name=None):
        """
        Inicializa um clone com meta-aprendizado.
        
        Args:
            algorithm (str): Algoritmo a ser usado 
            vectorizer (str): Tipo de vetorização
            hyperparams (dict): Hiperparâmetros para o algoritmo
            mutation_rates (dict): Taxas de mutação para diferentes aspectos
            meta_strategies (dict): Estratégias de meta-aprendizado
            name (str): Nome do clone
        """
        # Inicializa com a classe pai (AdaptiveClone)
        super().__init__(algorithm, vectorizer, hyperparams, mutation_rates, name)
        
        # Estratégias de meta-aprendizado
        self.meta_strategies = meta_strategies or {
            'feature_selection': {
                'active': False,           # Se está ativado
                'threshold': 0.01,         # Limiar para seleção de features
                'min_features': 100,       # Mínimo de features
                'max_features': 5000       # Máximo de features
            },
            'error_analysis': {
                'active': False,           # Se está ativado
                'sample_size': 0.2,        # Proporção de amostras para analisar
                'error_threshold': 0.3     # Limiar para considerar como erro frequente
            },
            'learning_rate_adjustment': {
                'active': False,           # Se está ativado
                'initial_lr': 0.1,         # Taxa de aprendizado inicial (para algoritmos que suportam)
                'decay_factor': 0.9,       # Fator de decaimento
                'min_lr': 0.001            # Taxa mínima
            },
            'early_stopping': {
                'active': False,           # Se está ativado
                'patience': 3,             # Número de iterações sem melhoria
                'min_delta': 0.001         # Melhoria mínima para considerar progresso
            },
            'sample_weighting': {
                'active': False,           # Se está ativado
                'weight_errors': 1.5,      # Peso para exemplos errôneos
                'reweight_interval': 2     # Reponderar após N iterações
            }
        }
        
        # Metadados específicos do meta-aprendizado
        self.meta_history = []  # Histórico de ajustes de meta-aprendizado
        
        # Atualiza os metadados do clone
        self.metadata.update({
            'type': 'meta',
            'meta_strategies': self.meta_strategies
        })
        
        self.log_event('created_meta')
        
    def log_meta_event(self, event_type: str, details: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Registra um evento de meta-aprendizado.
        
        Args:
            event_type: Tipo do evento ('strategy_adjusted', 'meta_evaluated', etc)
            details: Detalhes do evento
            
        Returns:
            Dict: Registro do evento
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details or {}
        }
        self.meta_history.append(event)
        
        # Também registra no histórico geral
        self.log_event(f'meta_{event_type}', details)
        
        return event
        
    def train(self, texts, labels):
        """
        Treina o clone com textos e seus sentimentos, aplicando meta-estratégias.
        
        Args:
            texts (list): Lista de textos para treinamento
            labels (list): Lista de rótulos (0=negativo, 1=positivo)
            
        Returns:
            dict: Métricas de treinamento
        """
        start_time = time.time()
        
        # Aplicar estratégias de meta-aprendizado antes do treinamento
        if self.meta_strategies['feature_selection']['active']:
            # Implementação simplificada de seleção de features 
            # Ajusta max_features do vetorizador com base na estratégia
            max_features = min(
                self.meta_strategies['feature_selection']['max_features'],
                max(self.meta_strategies['feature_selection']['min_features'],
                    int(len(texts) * 0.5))  # Heurística simples baseada no tamanho dos dados
            )
            
            # Cria um novo vetorizador com os parâmetros ajustados
            if self.vectorizer_name == 'tfidf':
                self.vectorizer = TfidfVectorizer(max_features=max_features)
            else:  # default: count
                self.vectorizer = CountVectorizer(max_features=max_features)
                
            self.log_meta_event('feature_selection_applied', {
                'max_features': max_features
            })
            
        # Transforma textos em vetores
        X_train = self.vectorizer.fit_transform(texts)
        
        # Aplicar estratégias de meta-aprendizado para o treinamento
        if self.algorithm_name == 'logistic' and self.meta_strategies['learning_rate_adjustment']['active']:
            # Ajusta a taxa de aprendizado (equivalente a ajustar C no caso da regressão logística)
            current_c = self.hyperparams.get('C', 1.0)
            new_c = current_c * self.meta_strategies['learning_rate_adjustment']['decay_factor']
            new_c = max(new_c, self.meta_strategies['learning_rate_adjustment']['min_lr'])
            
            self.hyperparams['C'] = new_c
            # Recria o modelo com o novo parâmetro
            self.model = self._create_algorithm()
            
            self.log_meta_event('learning_rate_adjusted', {
                'old_c': current_c,
                'new_c': new_c
            })
            
        # Aplica pesos de amostra se ativado e se temos histórico de erros anteriores
        sample_weights = None
        if self.meta_strategies['sample_weighting']['active'] and len(self.meta_history) > 0:
            # Simplificação: usa pesos uniformes para esta implementação
            sample_weights = np.ones(len(texts))
            
            # Se tivermos informações de erros anteriores, podemos usar
            for event in self.meta_history:
                if event['type'] == 'error_analysis_completed' and 'error_indices' in event['details']:
                    error_indices = event['details']['error_indices']
                    # Aumenta o peso das amostras que foram erroneamente classificadas
                    for idx in error_indices:
                        if 0 <= idx < len(sample_weights):
                            sample_weights[idx] = self.meta_strategies['sample_weighting']['weight_errors']
                    
                    self.log_meta_event('sample_weighting_applied', {
                        'weighted_samples': len(error_indices),
                        'weight_value': self.meta_strategies['sample_weighting']['weight_errors']
                    })
                    break
        
        # Treina o modelo com os pesos de amostra, se disponíveis
        if sample_weights is not None and hasattr(self.model, 'fit') and 'sample_weight' in self.model.fit.__code__.co_varnames:
            self.model.fit(X_train, labels, sample_weight=sample_weights)
        else:
            self.model.fit(X_train, labels)
        
        # Realiza análise de erro se ativada
        if self.meta_strategies['error_analysis']['active']:
            # Faz predições nos dados de treinamento
            train_preds = self.model.predict(X_train)
            
            # Identifica erros
            error_indices = [i for i, (true, pred) in enumerate(zip(labels, train_preds)) if true != pred]
            error_rate = len(error_indices) / len(labels)
            
            self.log_meta_event('error_analysis_completed', {
                'error_rate': error_rate,
                'error_count': len(error_indices),
                'error_indices': error_indices
            })
            
            # Ajusta estratégias com base na análise de erros
            if error_rate > self.meta_strategies['error_analysis']['error_threshold']:
                # Alta taxa de erro: ativa seleção de features e ponderação de amostras
                self.meta_strategies['feature_selection']['active'] = True
                self.meta_strategies['sample_weighting']['active'] = True
                
                self.log_meta_event('strategy_adjusted', {
                    'feature_selection': True,
                    'sample_weighting': True,
                    'reason': 'high_error_rate'
                })
        
        # Calcula tempo de treinamento
        self.training_time = time.time() - start_time
        
        # Registra evento
        metrics = {
            'training_time': self.training_time,
            'train_samples': len(texts),
            'meta_strategies_active': {k: v['active'] for k, v in self.meta_strategies.items()}
        }
        self.log_event('trained', metrics)
        
        return metrics
    
    def evaluate(self, texts, true_labels):
        """
        Avalia o desempenho do clone e ajusta suas meta-estratégias.
        
        Args:
            texts (list): Textos para avaliação
            true_labels (list): Rótulos verdadeiros
            
        Returns:
            dict: Métricas de desempenho
        """
        # Obtem métricas usando a implementação da classe pai
        metrics = super().evaluate(texts, true_labels)
        
        # Com base no desempenho, ajusta as meta-estratégias
        if metrics['accuracy'] < 0.7:
            # Desempenho ruim: ativa mais estratégias de meta-aprendizado
            self.meta_strategies['feature_selection']['active'] = True
            self.meta_strategies['error_analysis']['active'] = True
            self.meta_strategies['sample_weighting']['active'] = True
            
            if self.algorithm_name == 'logistic':
                self.meta_strategies['learning_rate_adjustment']['active'] = True
                
            self.log_meta_event('strategies_adjusted', {
                'reason': 'low_accuracy',
                'accuracy': metrics['accuracy'],
                'new_strategies': {k: v['active'] for k, v in self.meta_strategies.items()}
            })
        elif metrics['accuracy'] > 0.9:
            # Desempenho excelente: desativa algumas estratégias para simplificar
            self.meta_strategies['sample_weighting']['active'] = False
            
            self.log_meta_event('strategies_adjusted', {
                'reason': 'high_accuracy',
                'accuracy': metrics['accuracy'],
                'new_strategies': {k: v['active'] for k, v in self.meta_strategies.items()}
            })
        
        return metrics
    
    def mutate(self):
        """
        Cria uma versão modificada deste clone com meta-aprendizado.
        
        Returns:
            MetaClone: Um novo clone com parâmetros e estratégias mutadas
        """
        # Obtem hiperparâmetros e taxas de mutação do método da classe pai
        new_mutation_rates = self._mutate_mutation_rates()
        
        # Determina as mutações do algoritmo e vetorizador
        if random.random() < new_mutation_rates['algorithm_change_rate']:
            new_algorithm = random.choice(self.algorithm_options)
        else:
            new_algorithm = self.algorithm_name
            
        if random.random() < new_mutation_rates['vectorizer_change_rate']:
            new_vectorizer = random.choice(['count', 'tfidf'])
        else:
            new_vectorizer = self.vectorizer_name
            
        # Muta hiperparâmetros com base no algoritmo escolhido
        new_hyperparams = self._mutate_hyperparams(
            new_algorithm, 
            dict(self.hyperparams),
            new_mutation_rates
        )
        
        # Muta estratégias de meta-aprendizado
        new_meta_strategies = self._mutate_meta_strategies()
        
        # Cria novo clone com parâmetros mutados
        new_clone = MetaClone(
            algorithm=new_algorithm,
            vectorizer=new_vectorizer,
            hyperparams=new_hyperparams,
            mutation_rates=new_mutation_rates,
            meta_strategies=new_meta_strategies,
            name=f"Meta-{self.id}-{str(random.randint(1000, 9999))}"
        )
        
        # Configura relação de parentesco
        new_clone.parent_ids = [self.id]
        new_clone.generation = self.generation + 1
        
        # Registra evento de mutação com detalhes
        new_clone.log_event('mutated', {
            'parent_id': self.id,
            'mutations': {
                'algorithm': new_algorithm if new_algorithm != self.algorithm_name else 'unchanged',
                'vectorizer': new_vectorizer if new_vectorizer != self.vectorizer_name else 'unchanged',
                'hyperparams': {k: v for k, v in new_hyperparams.items() 
                               if k not in self.hyperparams or v != self.hyperparams[k]},
                'mutation_rates': {k: v for k, v in new_mutation_rates.items() 
                                  if v != self.mutation_rates.get(k)},
                'meta_strategies': {k: v for k, v in new_meta_strategies.items() 
                                   if json.dumps(v) != json.dumps(self.meta_strategies.get(k, {}))}
            }
        })
        
        return new_clone
    
    def _mutate_meta_strategies(self):
        """
        Muta as estratégias de meta-aprendizado.
        
        Returns:
            dict: Novas estratégias de meta-aprendizado
        """
        new_strategies = dict(self.meta_strategies)
        
        # Para cada estratégia, há uma chance de mutação
        for strategy_name, strategy in new_strategies.items():
            # 30% de chance de mudar o estado de ativação
            if random.random() < 0.3:
                strategy['active'] = not strategy['active']
                
            # Para cada parâmetro numérico, aplica uma pequena mutação
            for param, value in strategy.items():
                if param != 'active' and isinstance(value, (int, float)):
                    # Aplica mutação gaussiana
                    if isinstance(value, int) and value > 0:
                        # Para valores inteiros, usa escala proporcional ao valor
                        mutation = random.normalvariate(0, max(1, value * 0.2))
                        strategy[param] = max(1, int(value + mutation))
                    elif isinstance(value, float) and value > 0:
                        # Para valores float, usa escala proporcional
                        mutation = random.normalvariate(0, value * 0.2)
                        strategy[param] = max(0.0001, value + mutation)
        
        return new_strategies
    
    def save_meta_insights(self, filepath: str = None) -> Dict:
        """
        Salva insights de meta-aprendizado em um arquivo JSON.
        
        Args:
            filepath: Caminho onde salvar o arquivo
            
        Returns:
            Dict: Insights de meta-aprendizado
        """
        # Compila insights sobre o aprendizado
        insights = {
            'clone_id': self.id,
            'clone_name': self.name,
            'generation': self.generation,
            'performance_score': self.performance_score,
            'current_meta_strategies': self.meta_strategies,
            'meta_history': self.meta_history,
            'strategy_effectiveness': {}
        }
        
        # Analisa a eficácia das estratégias de meta-aprendizado
        strategy_activations = {}
        performance_by_strategy = {}
        
        for event in self.history:
            if event['type'] == 'evaluated' and 'details' in event and 'accuracy' in event['details']:
                # Encontra quais estratégias estavam ativas no momento
                active_strategies = set()
                
                # Busca as estratégias ativas mais recentes antes dessa avaliação
                for meta_event in sorted(self.meta_history, key=lambda x: x['timestamp'], reverse=True):
                    if meta_event['timestamp'] < event['timestamp'] and 'details' in meta_event:
                        if 'new_strategies' in meta_event['details']:
                            for strategy, active in meta_event['details']['new_strategies'].items():
                                if active and strategy not in active_strategies:
                                    active_strategies.add(strategy)
                            break
                
                # Registra desempenho para cada estratégia ativa
                for strategy in active_strategies:
                    if strategy not in performance_by_strategy:
                        performance_by_strategy[strategy] = []
                    
                    performance_by_strategy[strategy].append(event['details']['accuracy'])
        
        # Calcula eficácia média das estratégias
        for strategy, performances in performance_by_strategy.items():
            if performances:
                insights['strategy_effectiveness'][strategy] = {
                    'mean_accuracy': np.mean(performances),
                    'std_accuracy': np.std(performances),
                    'sample_count': len(performances)
                }
        
        # Salva os insights se um caminho for fornecido
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(insights, f, indent=2)
        
        return insights 
