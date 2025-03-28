from mestre_dos_clones.clones.base_clone import BaseClone
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import random

class RegressionClone(BaseClone):
    """Clone especializado em regressão para previsão de valores contínuos."""
    
    AVAILABLE_ALGORITHMS = {
        'random_forest_regressor': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'svr': SVR
    }

    def __init__(self, algorithm_name='random_forest_regressor', **kwargs):
        # Extrair parâmetros específicos da classe BaseClone
        base_kwargs = {k: v for k, v in kwargs.items() 
                      if k in ['name', 'parent_ids', 'generation']}
        
        # Chamar o construtor da classe base apenas com os parâmetros que ela aceita
        super().__init__(**base_kwargs)
        
        # Atributos específicos dessa classe
        self.algorithm_name = algorithm_name
        self.algorithm = None
        self.vectorizer = None
        self.vectorizer_name = None
        self.performance_score = None
        self.seed = kwargs.get('seed', 42)
        self._setup_algorithm()
    
    def _setup_algorithm(self):
        """Configura o algoritmo de regressão com parâmetros adequados."""
        if self.algorithm_name == 'random_forest_regressor':
            self.algorithm = RandomForestRegressor(
                n_estimators=100,
                random_state=self.seed
            )
        elif self.algorithm_name == 'gradient_boosting':
            self.algorithm = GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.seed
            )
        elif self.algorithm_name == 'linear_regression':
            self.algorithm = LinearRegression()
        elif self.algorithm_name == 'ridge':
            self.algorithm = Ridge(alpha=1.0, random_state=self.seed)
        elif self.algorithm_name == 'lasso':
            self.algorithm = Lasso(alpha=0.1, random_state=self.seed)
        elif self.algorithm_name == 'svr':
            self.algorithm = SVR(kernel='rbf', C=1.0, epsilon=0.1)
        else:
            # Fallback para RandomForest
            self.algorithm_name = 'random_forest_regressor'
            self.algorithm = RandomForestRegressor(
                n_estimators=100,
                random_state=self.seed
            )
    
    def train(self, X, y):
        """Treina o modelo de regressão."""
        # X já deve estar no formato adequado para o algoritmo
        self.algorithm.fit(X, y)
        self.log_event('trained')
    
    def predict(self, X):
        """Faz previsões de valores contínuos."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de fazer previsões.")
        
        return self.algorithm.predict(X)
    
    def evaluate(self, X, y):
        """Avalia o desempenho do modelo de regressão."""
        if self.algorithm is None:
            raise ValueError("O clone precisa ser treinado antes de ser avaliado.")
        
        y_pred = self.predict(X)
        
        # Calcula métricas de avaliação
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Normaliza para uma pontuação entre 0 e 1 (onde 1 é melhor)
        # Usando R² como métrica principal, com uma penalidade para MSE alto
        # Assumindo que quanto menor o MSE, melhor (independentemente da escala)
        mse_norm = np.exp(-mse / 10)  # Normaliza MSE para um valor entre 0 e 1
        
        # Combina as métricas, dando mais peso ao R²
        self.performance_score = 0.7 * max(0, r2) + 0.3 * mse_norm
        
        return self.performance_score
    
    def mutate(self):
        """Gera uma versão mutada do clone."""
        # Herdado de características do clone original
        mutation_strength = random.uniform(0.1, 0.5)
        
        # Escolhe aleatoriamente um novo algoritmo com probabilidade baixa
        if random.random() < mutation_strength:
            algorithms = list(self.AVAILABLE_ALGORITHMS.keys())
            algorithms.remove(self.algorithm_name)
            new_algorithm = random.choice(algorithms)
        else:
            new_algorithm = self.algorithm_name
        
        # Cria um novo clone com o algoritmo possivelmente alterado
        mutated_clone = RegressionClone(
            algorithm_name=new_algorithm,
            name=f"Regression-{self.id[:8]}-{random.randint(1000, 9999)}",
            parent_ids=[self.id],
            generation=self.generation + 1,
            seed=random.randint(1, 10000)
        )
        
        # Registra o evento de mutação
        self.log_event('mutated', {'child_id': mutated_clone.id})
        
        return mutated_clone 
