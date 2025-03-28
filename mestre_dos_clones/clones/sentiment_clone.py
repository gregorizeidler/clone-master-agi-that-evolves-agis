import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import random

from mestre_dos_clones.clones.base_clone import BaseClone


class SentimentClone(BaseClone):
    """
    Clone especializado em classificação de sentimentos em textos.
    Pode usar diferentes algoritmos e representações de texto.
    """
    
    def __init__(self, 
                 algorithm='naive_bayes', 
                 vectorizer='count',
                 hyperparams=None,
                 name=None):
        """
        Inicializa um clone de análise de sentimentos.
        
        Args:
            algorithm (str): Algoritmo a ser usado ('naive_bayes', 'logistic', 'svm')
            vectorizer (str): Tipo de vetorização ('count', 'tfidf')
            hyperparams (dict, optional): Hiperparâmetros para o algoritmo
            name (str, optional): Nome do clone
        """
        super().__init__(name)
        
        self.algorithm_name = algorithm
        self.vectorizer_name = vectorizer
        self.hyperparams = hyperparams or {}
        
        # Inicializa o vetorizador
        if vectorizer == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=5000)
        else:  # default: count
            self.vectorizer = CountVectorizer(max_features=5000)
            
        # Inicializa o classificador
        self.model = self._create_algorithm()
        
        # Metadados do clone
        self.metadata = {
            'algorithm': algorithm,
            'vectorizer': vectorizer,
            'hyperparams': self.hyperparams
        }
        
        self.log_event('created')
        
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
        else:  # default: naive_bayes
            return MultinomialNB(
                **{k: v for k, v in self.hyperparams.items() 
                   if k in ['alpha', 'fit_prior']}
            )
    
    def train(self, texts, labels):
        """
        Treina o clone com textos e seus sentimentos.
        
        Args:
            texts (list): Lista de textos para treinamento
            labels (list): Lista de rótulos (0=negativo, 1=positivo)
            
        Returns:
            dict: Métricas de treinamento
        """
        start_time = time.time()
        
        # Transforma textos em vetores
        X_train = self.vectorizer.fit_transform(texts)
        
        # Treina o modelo
        self.model.fit(X_train, labels)
        
        # Calcula tempo de treinamento
        self.training_time = time.time() - start_time
        
        # Registra evento
        metrics = {
            'training_time': self.training_time,
            'train_samples': len(texts)
        }
        self.log_event('trained', metrics)
        
        return metrics
    
    def predict(self, texts):
        """
        Classifica textos como positivos ou negativos.
        
        Args:
            texts (list): Lista de textos para classificação
            
        Returns:
            list: Lista de predições (0=negativo, 1=positivo)
        """
        X = self.vectorizer.transform(texts)
        predictions = self.model.predict(X)
        
        self.log_event('predicted', {'samples': len(texts)})
        return predictions
    
    def evaluate(self, texts, true_labels):
        """
        Avalia o desempenho do clone em um conjunto de dados.
        
        Args:
            texts (list): Textos para avaliação
            true_labels (list): Rótulos verdadeiros
            
        Returns:
            dict: Métricas de desempenho
        """
        predictions = self.predict(texts)
        
        # Calcula métricas
        metrics = {
            'accuracy': accuracy_score(true_labels, predictions),
            'precision': precision_score(true_labels, predictions, zero_division=0),
            'recall': recall_score(true_labels, predictions, zero_division=0),
            'f1': f1_score(true_labels, predictions, zero_division=0),
        }
        
        # Atualiza pontuação geral do clone (usando F1 como principal métrica)
        self.performance_score = metrics['f1']
        
        self.log_event('evaluated', metrics)
        return metrics
    
    def mutate(self):
        """
        Cria uma versão modificada deste clone com pequenas alterações.
        
        Returns:
            SentimentClone: Um novo clone com parâmetros mutados
        """
        # Possíveis mutações
        algorithm_options = ['naive_bayes', 'logistic', 'svm']
        vectorizer_options = ['count', 'tfidf']
        
        # 20% de chance de mudar completamente o algoritmo
        if random.random() < 0.2:
            new_algorithm = random.choice(algorithm_options)
        else:
            new_algorithm = self.algorithm_name
            
        # 30% de chance de mudar o vetorizador
        if random.random() < 0.3:
            new_vectorizer = random.choice(vectorizer_options)
        else:
            new_vectorizer = self.vectorizer_name
            
        # Muta hiperparâmetros baseado no algoritmo escolhido
        new_hyperparams = dict(self.hyperparams)  # Cópia dos hiperparâmetros atuais
        
        if new_algorithm == 'naive_bayes':
            # Altera alpha do Naive Bayes
            new_hyperparams['alpha'] = max(0.01, np.random.normal(
                loc=new_hyperparams.get('alpha', 1.0),
                scale=0.3
            ))
            
        elif new_algorithm == 'logistic':
            # Altera C da Regressão Logística
            new_hyperparams['C'] = max(0.1, np.random.normal(
                loc=new_hyperparams.get('C', 1.0),
                scale=0.5
            ))
            new_hyperparams['max_iter'] = max(100, int(
                np.random.normal(
                    loc=new_hyperparams.get('max_iter', 100),
                    scale=20
                )
            ))
            
        elif new_algorithm == 'svm':
            # Altera C do SVM
            new_hyperparams['C'] = max(0.1, np.random.normal(
                loc=new_hyperparams.get('C', 1.0),
                scale=0.5
            ))
            
        # Cria novo clone com parâmetros mutados
        new_clone = SentimentClone(
            algorithm=new_algorithm,
            vectorizer=new_vectorizer,
            hyperparams=new_hyperparams,
            name=f"Mutant-{self.id}-{str(random.randint(1000, 9999))}"
        )
        
        # Configura relação de parentesco
        new_clone.parent_ids = [self.id]
        new_clone.generation = self.generation + 1
        
        new_clone.log_event('mutated', {
            'parent_id': self.id,
            'mutations': {
                'algorithm': new_algorithm if new_algorithm != self.algorithm_name else 'unchanged',
                'vectorizer': new_vectorizer if new_vectorizer != self.vectorizer_name else 'unchanged',
                'hyperparams': {k: v for k, v in new_hyperparams.items() if k not in self.hyperparams or v != self.hyperparams[k]}
            }
        })
        
        return new_clone
