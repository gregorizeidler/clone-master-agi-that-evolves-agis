from mestre_dos_clones.clones.base_clone import BaseClone
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import random

class ClusteringClone(BaseClone):
    """Clone especializado em agrupamento não supervisionado."""
    
    AVAILABLE_ALGORITHMS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
        'gaussian_mixture': GaussianMixture
    }
    
    def __init__(self, algorithm_name='kmeans', **kwargs):
        # Extrair parâmetros específicos da classe BaseClone
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['name', 'parent_ids', 'generation']}
        
        # Chamar o construtor da classe base apenas com os parâmetros que ela aceita
        super().__init__(**base_kwargs)
        
        # Atributos específicos dessa classe
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.n_clusters = kwargs.get('n_clusters', 3)
        self.performance_score = None
        self.seed = kwargs.get('seed', 42)
        self._setup_algorithm()
    
    def _setup_algorithm(self):
        """Configura o algoritmo de clustering com parâmetros adequados."""
        if self.algorithm_name == 'kmeans':
            self.algorithm = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed
            )
        elif self.algorithm_name == 'dbscan':
            self.algorithm = DBSCAN(
                eps=0.5,
                min_samples=5
            )
        elif self.algorithm_name == 'agglomerative':
            self.algorithm = AgglomerativeClustering(
                n_clusters=self.n_clusters
            )
        elif self.algorithm_name == 'gaussian_mixture':
            self.algorithm = GaussianMixture(
                n_components=self.n_clusters,
                random_state=self.seed
            )
        else:
            # Fallback para KMeans
            self.algorithm_name = 'kmeans'
            self.algorithm = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.seed
            )
    
    def train(self, X, y=None):
        """Treina o modelo de clustering (y é ignorado)."""
        # X já deve estar no formato adequado para o algoritmo
        self.algorithm.fit(X)
        self.log_event('trained')
    
    def predict(self, X):
        """Atribui clusters a novos dados."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de fazer previsões.")
        
        if hasattr(self.algorithm, 'predict'):
            return self.algorithm.predict(X)
        elif hasattr(self.algorithm, 'fit_predict'):
            # Para algoritmos como DBSCAN que usam fit_predict
            return self.algorithm.fit_predict(X)
    
    def evaluate(self, X, y=None):
        """Avalia a qualidade do clustering."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de ser avaliado.")
        
        # Obter as previsões de cluster
        if hasattr(self.algorithm, 'labels_'):
            labels = self.algorithm.labels_
        else:
            labels = self.predict(X)
        
        # Calcular métricas apenas se temos mais de um cluster
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels[unique_labels != -1])  # -1 é ruído em DBSCAN
        
        if n_clusters > 1 and n_clusters < len(X) - 1:
            try:
                silhouette = silhouette_score(X, labels)
                calinski = calinski_harabasz_score(X, labels)
                # Média das métricas (normalizadas)
                self.performance_score = (silhouette + 1) / 2  # silhouette vai de -1 a 1
            except:
                # Fallback se o cálculo das métricas falhar
                self.performance_score = 0.5
        else:
            self.performance_score = 0.5
        
        return self.performance_score
    
    def mutate(self):
        """Gera uma versão mutada do clone."""
        # Herdado de características do clone original
        mutation_strength = random.uniform(0.1, 0.5)
        
        # Escolher novo algoritmo ou manter o atual
        if random.random() < mutation_strength:
            algorithms = list(self.AVAILABLE_ALGORITHMS.keys())
            algorithms.remove(self.algorithm_name)
            new_algorithm = random.choice(algorithms)
        else:
            new_algorithm = self.algorithm_name
        
        # Mutar o número de clusters
        new_n_clusters = max(2, self.n_clusters + random.randint(-2, 2))
        
        # Cria um novo clone com as alterações
        mutated_clone = ClusteringClone(
            algorithm_name=new_algorithm,
            n_clusters=new_n_clusters,
            name=f"Cluster-{self.id[:8]}-{random.randint(1000, 9999)}",
            parent_ids=[self.id],
            generation=self.generation + 1,
            seed=random.randint(1, 10000)
        )
        
        # Registra o evento de mutação
        self.log_event('mutated', {'child_id': mutated_clone.id})
        
        return mutated_clone 
