from mestre_dos_clones.clones.base_clone import BaseClone
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.manifold import TSNE, Isomap
import random

class DimensionClone(BaseClone):
    """Clone especializado em redução de dimensionalidade."""
    
    AVAILABLE_ALGORITHMS = {
        'pca': PCA,
        'truncated_svd': TruncatedSVD,
        'tsne': TSNE,
        'isomap': Isomap,
        'fast_ica': FastICA
    }
    
    def __init__(self, algorithm_name='pca', **kwargs):
        # Extrair parâmetros específicos da classe BaseClone
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['name', 'parent_ids', 'generation']}
        
        # Chamar o construtor da classe base apenas com os parâmetros que ela aceita
        super().__init__(**base_kwargs)
        
        # Atributos específicos dessa classe
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.n_components = kwargs.get('n_components', 2)
        self.performance_score = None
        self.seed = kwargs.get('seed', 42)
        self._setup_algorithm()
    
    def _setup_algorithm(self):
        """Configura o algoritmo de redução de dimensionalidade."""
        if self.algorithm_name == 'pca':
            self.algorithm = PCA(
                n_components=self.n_components,
                random_state=self.seed
            )
        elif self.algorithm_name == 'truncated_svd':
            self.algorithm = TruncatedSVD(
                n_components=self.n_components,
                random_state=self.seed
            )
        elif self.algorithm_name == 'tsne':
            self.algorithm = TSNE(
                n_components=self.n_components,
                random_state=self.seed
            )
        elif self.algorithm_name == 'isomap':
            self.algorithm = Isomap(
                n_components=self.n_components
            )
        elif self.algorithm_name == 'fast_ica':
            self.algorithm = FastICA(
                n_components=self.n_components,
                random_state=self.seed
            )
        else:
            # Fallback para PCA
            self.algorithm_name = 'pca'
            self.algorithm = PCA(
                n_components=self.n_components,
                random_state=self.seed
            )
    
    def train(self, X, y=None):
        """Treina o modelo de redução de dimensionalidade (y é ignorado)."""
        self.algorithm.fit(X)
        self.log_event('trained')
    
    def transform(self, X):
        """Transforma os dados para o espaço de dimensão reduzida."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de transformar dados.")
        
        return self.algorithm.transform(X)
    
    def predict(self, X):
        """Implementa a interface de previsão usando transform."""
        return self.transform(X)
    
    def evaluate(self, X, y=None):
        """Avalia a qualidade da redução de dimensionalidade."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de ser avaliado.")
        
        # Para PCA e SVD, podemos usar a variância explicada
        if hasattr(self.algorithm, 'explained_variance_ratio_'):
            # Soma da variância explicada
            self.performance_score = np.sum(self.algorithm.explained_variance_ratio_)
        else:
            # Para outros métodos como t-SNE que não têm métrica direta
            # Usamos um valor fixo como placeholder
            self.performance_score = 0.7
        
        return self.performance_score
    
    def mutate(self):
        """Gera uma versão mutada do clone."""
        mutation_strength = random.uniform(0.1, 0.5)
        
        # Escolher novo algoritmo ou manter o atual
        if random.random() < mutation_strength:
            algorithms = list(self.AVAILABLE_ALGORITHMS.keys())
            algorithms.remove(self.algorithm_name)
            new_algorithm = random.choice(algorithms)
        else:
            new_algorithm = self.algorithm_name
        
        # Mutar o número de componentes
        new_n_components = max(2, self.n_components + random.randint(-1, 1))
        
        # Cria um novo clone com as alterações
        mutated_clone = DimensionClone(
            algorithm_name=new_algorithm,
            n_components=new_n_components,
            name=f"Dimension-{self.id[:8]}-{random.randint(1000, 9999)}",
            parent_ids=[self.id],
            generation=self.generation + 1,
            seed=random.randint(1, 10000)
        )
        
        # Registra o evento de mutação
        self.log_event('mutated', {'child_id': mutated_clone.id})
        
        return mutated_clone 
