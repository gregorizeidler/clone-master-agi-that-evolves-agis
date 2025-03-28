import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import random

from mestre_dos_clones.clones.sentiment_clone import SentimentClone


class AdaptiveClone(SentimentClone):
    """
    Clone com capacidade de auto-adaptação de parâmetros evolutivos.
    Este clone herda as capacidades do SentimentClone, mas adiciona mecanismos
    para ajustar automaticamente seus próprios parâmetros de mutação durante a evolução.
    """
    
    def __init__(self, 
                 algorithm='naive_bayes', 
                 vectorizer='count',
                 hyperparams=None,
                 mutation_rates=None,
                 name=None):
        """
        Inicializa um clone adaptativo.
        
        Args:
            algorithm (str): Algoritmo a ser usado ('naive_bayes', 'logistic', 'svm', 'random_forest')
            vectorizer (str): Tipo de vetorização ('count', 'tfidf')
            hyperparams (dict, optional): Hiperparâmetros para o algoritmo
            mutation_rates (dict, optional): Taxas de mutação para diferentes aspectos do clone
            name (str, optional): Nome do clone
        """
        # Inicializa com a classe pai
        super().__init__(algorithm, vectorizer, hyperparams, name)
        
        # Parâmetros de auto-adaptação
        self.mutation_rates = mutation_rates or {
            'algorithm_change_rate': 0.2,   # Probabilidade de mudar o algoritmo
            'vectorizer_change_rate': 0.3,  # Probabilidade de mudar o vetorizador
            'param_mutation_strength': 0.3, # Força da mutação dos hiperparâmetros
            'param_mutation_rate': 0.5      # Probabilidade de mutar cada hiperparâmetro
        }
        
        # Expande as opções de algoritmos
        self.algorithm_options = ['naive_bayes', 'logistic', 'svm', 'random_forest']
        
        # Atualiza metadata
        self.metadata.update({
            'type': 'adaptive',
            'mutation_rates': self.mutation_rates
        })
        
        self.log_event('created_adaptive')
    
    def _create_algorithm(self):
        """Cria o algoritmo de classificação com base nos parâmetros."""
        if self.algorithm_name == 'logistic':
            return LogisticRegression(
                **{k: v for k, v in self.hyperparams.items() 
                   if k in ['C', 'max_iter', 'solver', 'penalty']}
            )
        elif self.algorithm_name == 'svm':
            return SVC(
                **{k: v for k, v in self.hyperparams.items() 
                   if k in ['C', 'kernel', 'gamma']}
            )
        elif self.algorithm_name == 'random_forest':
            return RandomForestClassifier(
                **{k: v for k, v in self.hyperparams.items()
                   if k in ['n_estimators', 'max_depth', 'min_samples_split']}
            )
        else:  # default: naive_bayes
            return MultinomialNB(
                **{k: v for k, v in self.hyperparams.items() 
                   if k in ['alpha', 'fit_prior']}
            )
    
    def mutate(self):
        """
        Cria uma versão modificada deste clone com auto-adaptação de parâmetros.
        
        Returns:
            AdaptiveClone: Um novo clone com parâmetros mutados
        """
        # Primeiro, mutamos as próprias taxas de mutação
        new_mutation_rates = self._mutate_mutation_rates()
        
        # Usa as novas taxas para determinar as mutações do clone
        # Determina se muda o algoritmo
        if random.random() < new_mutation_rates['algorithm_change_rate']:
            new_algorithm = random.choice(self.algorithm_options)
        else:
            new_algorithm = self.algorithm_name
            
        # Determina se muda o vetorizador
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
        
        # Cria novo clone com parâmetros mutados
        new_clone = AdaptiveClone(
            algorithm=new_algorithm,
            vectorizer=new_vectorizer,
            hyperparams=new_hyperparams,
            mutation_rates=new_mutation_rates,
            name=f"Adaptive-{self.id}-{str(random.randint(1000, 9999))}"
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
                                  if v != self.mutation_rates.get(k)}
            }
        })
        
        return new_clone
        
    def _mutate_mutation_rates(self):
        """
        Muta as próprias taxas de mutação (auto-adaptação).
        
        Returns:
            dict: Novas taxas de mutação
        """
        new_rates = dict(self.mutation_rates)
        
        # Para cada taxa de mutação, aplicamos uma pequena mutação
        for rate_name, rate_value in new_rates.items():
            # Aplicamos uma mutação gaussiana com escala pequena
            mutation = np.random.normal(0, 0.05)
            
            # Aplicamos a mutação e garantimos que fique entre 0.05 e 0.95
            new_rates[rate_name] = max(0.05, min(0.95, rate_value + mutation))
            
        return new_rates
    
    def _mutate_hyperparams(self, algorithm, current_params, mutation_rates):
        """
        Muta os hiperparâmetros com base no algoritmo e nas taxas de mutação.
        
        Args:
            algorithm: Nome do algoritmo
            current_params: Parâmetros atuais
            mutation_rates: Taxas de mutação
            
        Returns:
            dict: Novos hiperparâmetros
        """
        new_params = dict(current_params)
        
        # Definições específicas para cada algoritmo
        if algorithm == 'naive_bayes':
            # Se o parâmetro não existe ou se decidimos mutá-lo
            if 'alpha' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('alpha', 1.0)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['alpha'] = max(0.01, np.random.normal(
                    loc=current,
                    scale=current * mutation_strength  # Força proporcional ao valor atual
                ))
                
        elif algorithm == 'logistic':
            if 'C' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('C', 1.0)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['C'] = max(0.1, np.random.normal(
                    loc=current,
                    scale=current * mutation_strength
                ))
                
            if 'max_iter' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('max_iter', 100)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['max_iter'] = max(50, int(
                    np.random.normal(
                        loc=current,
                        scale=current * mutation_strength
                    )
                ))
                
        elif algorithm == 'svm':
            if 'C' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('C', 1.0)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['C'] = max(0.1, np.random.normal(
                    loc=current,
                    scale=current * mutation_strength
                ))
                
            # 20% de chance de mudar o kernel se não existir ou se decidirmos mutá-lo
            if 'kernel' not in new_params or random.random() < 0.2:
                kernels = ['linear', 'rbf', 'poly']
                current = new_params.get('kernel', 'rbf')
                # Remove o kernel atual da lista para garantir uma mudança
                if current in kernels:
                    kernels.remove(current)
                new_params['kernel'] = random.choice(kernels)
                
        elif algorithm == 'random_forest':
            if 'n_estimators' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('n_estimators', 100)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['n_estimators'] = max(10, int(
                    np.random.normal(
                        loc=current,
                        scale=current * mutation_strength
                    )
                ))
                
            if 'max_depth' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('max_depth', 10)
                mutation_strength = mutation_rates['param_mutation_strength']
                # None é um valor válido (sem limite de profundidade)
                if current is not None:
                    new_params['max_depth'] = max(2, int(
                        np.random.normal(
                            loc=current,
                            scale=current * mutation_strength
                        )
                    ))
                # 10% de chance de mudar para None se tiver um valor atual
                elif random.random() < 0.1:
                    new_params['max_depth'] = None
                    
            if 'min_samples_split' not in new_params or random.random() < mutation_rates['param_mutation_rate']:
                current = new_params.get('min_samples_split', 2)
                mutation_strength = mutation_rates['param_mutation_strength']
                new_params['min_samples_split'] = max(2, int(
                    np.random.normal(
                        loc=current,
                        scale=current * mutation_strength * 0.5
                    )
                ))
                
        return new_params 
