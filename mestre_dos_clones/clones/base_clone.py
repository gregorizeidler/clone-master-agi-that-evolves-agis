from abc import ABC, abstractmethod
import pickle
import time
import os
import uuid


class BaseClone(ABC):
    """
    Classe base para todos os clones (AGIs) criados pelo sistema.
    Define a interface e funcionalidades comuns a todos os clones.
    """
    
    def __init__(self, name=None, parent_ids=None, generation=1):
        """
        Inicializa um novo clone com identificador único.
        
        Args:
            name (str, optional): Nome personalizado para o clone.
            parent_ids (list, optional): Lista de IDs dos clones pais.
            generation (int, optional): Geração do clone.
        """
        self.id = str(uuid.uuid4())[:8]  # ID único curto
        self.name = name or f"Clone-{self.id}"
        self.creation_time = time.time()
        self.training_time = 0
        self.performance_score = None
        self.generation = generation
        self.parent_ids = parent_ids or []
        self.metadata = {}
        self.history = []
        
    def log_event(self, event_type, details=None):
        """
        Registra um evento na história do clone.
        
        Args:
            event_type (str): Tipo de evento ('created', 'trained', 'evaluated', etc)
            details (dict, optional): Detalhes adicionais sobre o evento
        """
        event = {
            'timestamp': time.time(),
            'type': event_type,
            'details': details or {}
        }
        self.history.append(event)
        return event
        
    @abstractmethod
    def train(self, data, labels):
        """
        Treina o clone com dados fornecidos.
        
        Args:
            data: Dados de entrada para treinamento
            labels: Rótulos correspondentes
            
        Returns:
            dict: Métricas de treinamento
        """
        pass
    
    @abstractmethod
    def predict(self, data):
        """
        Realiza predições com o clone treinado.
        
        Args:
            data: Dados para predição
            
        Returns:
            Predições geradas pelo clone
        """
        pass
    
    def save(self, directory="generated"):
        """
        Salva o clone em disco.
        
        Args:
            directory (str): Diretório onde o clone será salvo
            
        Returns:
            str: Caminho onde o clone foi salvo
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        filepath = os.path.join(directory, f"{self.name}.pkl")
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        self.log_event('saved', {'path': filepath})
        return filepath
    
    @classmethod
    def load(cls, filepath):
        """
        Carrega um clone de um arquivo.
        
        Args:
            filepath (str): Caminho para o arquivo do clone
            
        Returns:
            BaseClone: Clone carregado do arquivo
        """
        with open(filepath, 'rb') as f:
            clone = pickle.load(f)
        
        clone.log_event('loaded', {'path': filepath})
        return clone
    
    def get_info(self):
        """
        Retorna informações sobre o clone.
        
        Returns:
            dict: Informações do clone
        """
        return {
            'id': self.id,
            'name': self.name,
            'creation_time': self.creation_time,
            'training_time': self.training_time,
            'performance_score': self.performance_score,
            'generation': self.generation,
            'parent_ids': self.parent_ids,
            'metadata': self.metadata,
            'type': self.__class__.__name__
        }
    
    def __str__(self):
        """Representação em string do clone."""
        score = f"{self.performance_score:.4f}" if self.performance_score is not None else "N/A"
        return f"{self.name} (Gen {self.generation}, Score: {score})"
