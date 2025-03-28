"""
Métricas e ferramentas para medir diversidade em populações de clones.

Este módulo fornece funções para quantificar a diversidade genética
e comportamental na população de clones, além de ferramentas de visualização.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import math
from collections import Counter
import os


def calculate_algorithm_diversity(clones: List) -> Dict[str, Any]:
    """
    Calcula métricas de diversidade baseadas nos algoritmos usados.
    
    Args:
        clones: Lista de clones para análise
        
    Returns:
        Dict: Métricas de diversidade de algoritmos
    """
    # Contagem de algoritmos
    algorithms = [clone.algorithm_name for clone in clones]
    algo_counts = Counter(algorithms)
    
    # Calcula o índice de diversidade de Shannon
    total = len(algorithms)
    shannon = 0
    
    for algo, count in algo_counts.items():
        p = count / total
        shannon -= p * math.log(p)
    
    # Calcula o índice de Simpson (1 - D)
    simpson = 1 - sum((count / total) ** 2 for count in algo_counts.values())
    
    return {
        'unique_algorithms': len(algo_counts),
        'algorithm_counts': dict(algo_counts),
        'shannon_index': shannon,
        'simpson_index': simpson,
        'evenness': shannon / math.log(max(len(algo_counts), 1))  # Normalizado para [0, 1]
    }


def calculate_hyperparameter_diversity(clones: List) -> Dict[str, Any]:
    """
    Calcula a diversidade de hiperparâmetros na população.
    
    Args:
        clones: Lista de clones para análise
        
    Returns:
        Dict: Métricas de diversidade de hiperparâmetros
    """
    # Coleta todos os hiperparâmetros
    all_hyperparams = {}
    
    for clone in clones:
        # Verifica se o clone tem o atributo hyperparams
        if not hasattr(clone, 'hyperparams'):
            # Para clones que não têm hyperparams, tenta extrair parâmetros relevantes
            if hasattr(clone, 'algorithm_name'):
                params_dict = {'algorithm': clone.algorithm_name}
                
                # Tenta extrair parâmetros específicos de cada tipo de clone
                if hasattr(clone, 'n_clusters'):
                    params_dict['n_clusters'] = clone.n_clusters
                if hasattr(clone, 'n_components'):
                    params_dict['n_components'] = clone.n_components
                if hasattr(clone, 'seed'):
                    params_dict['seed'] = clone.seed
            else:
                # Skip se não conseguir extrair parâmetros
                continue
        else:
            # Use hyperparams normalmente se estiver disponível
            params_dict = clone.hyperparams
        
        # Adiciona os parâmetros encontrados
        for param, value in params_dict.items():
            if param not in all_hyperparams:
                all_hyperparams[param] = []
            all_hyperparams[param].append(value)
    
    # Calcula estatísticas para cada hiperparâmetro numérico
    hyperparam_stats = {}
    
    for param, values in all_hyperparams.items():
        # Processa apenas hiperparâmetros numéricos
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if numeric_values:
            hyperparam_stats[param] = {
                'min': min(numeric_values),
                'max': max(numeric_values),
                'mean': np.mean(numeric_values),
                'std': np.std(numeric_values),
                'coefficient_of_variation': np.std(numeric_values) / np.mean(numeric_values) if np.mean(numeric_values) != 0 else 0,
                'unique_values': len(set(numeric_values)),
                'count': len(numeric_values)
            }
    
    # Calcula um índice geral de diversidade baseado na variação total
    avg_cv = np.mean([stats['coefficient_of_variation'] 
                     for stats in hyperparam_stats.values()
                     if not np.isnan(stats['coefficient_of_variation'])])
    
    return {
        'parameter_stats': hyperparam_stats,
        'average_coefficient_of_variation': avg_cv if not np.isnan(avg_cv) else 0.0,
        'total_unique_parameters': len(all_hyperparams)
    }


def calculate_performance_diversity(clones: List) -> Dict[str, Any]:
    """
    Calcula a diversidade de desempenho na população.
    
    Args:
        clones: Lista de clones para análise
        
    Returns:
        Dict: Métricas de diversidade de desempenho
    """
    # Extrai pontuações de desempenho, ignorando None
    scores = [clone.performance_score for clone in clones 
              if clone.performance_score is not None]
    
    if not scores:
        return {
            'score_count': 0,
            'score_range': 0,
            'score_std': 0,
            'score_mean': 0,
            'score_cv': 0
        }
    
    # Calcula estatísticas básicas
    score_min = min(scores)
    score_max = max(scores)
    score_mean = np.mean(scores)
    score_std = np.std(scores)
    score_cv = score_std / score_mean if score_mean != 0 else 0
    
    # Agrupa em bins para análise de distribuição
    hist, bins = np.histogram(scores, bins=5)
    bin_counts = {f"{bins[i]:.2f}-{bins[i+1]:.2f}": int(hist[i]) for i in range(len(hist))}
    
    return {
        'score_count': len(scores),
        'score_min': score_min,
        'score_max': score_max,
        'score_range': score_max - score_min,
        'score_mean': score_mean,
        'score_std': score_std,
        'score_cv': score_cv,  # Coeficiente de variação
        'score_distribution': bin_counts
    }


def calculate_generation_diversity(clones: List) -> Dict[str, Any]:
    """
    Calcula a diversidade de gerações na população.
    
    Args:
        clones: Lista de clones para análise
        
    Returns:
        Dict: Métricas de diversidade de gerações
    """
    generations = [clone.generation for clone in clones]
    gen_counts = Counter(generations)
    
    return {
        'generation_range': max(generations) - min(generations) if generations else 0,
        'generation_counts': dict(gen_counts),
        'unique_generations': len(gen_counts)
    }


def calculate_clone_diversity(clones: List) -> Dict[str, Any]:
    """
    Calcula métricas de diversidade gerais para uma população de clones.
    
    Args:
        clones: Lista de clones para análise
        
    Returns:
        Dict: Métricas de diversidade
    """
    # Combina as diferentes dimensões de diversidade
    results = {
        'population_size': len(clones),
        'algorithm_diversity': calculate_algorithm_diversity(clones),
        'hyperparameter_diversity': calculate_hyperparameter_diversity(clones),
        'performance_diversity': calculate_performance_diversity(clones),
        'generation_diversity': calculate_generation_diversity(clones)
    }
    
    # Calcula um índice de diversidade geral
    algo_diversity = results['algorithm_diversity']['simpson_index']
    hyperparam_diversity = results['hyperparameter_diversity']['average_coefficient_of_variation']
    perf_diversity = results['performance_diversity']['score_cv']
    
    # Média ponderada dos diferentes aspectos da diversidade
    # Normaliza primeiro o hyperparam_diversity que pode ser maior que 1
    norm_hyperparam = min(1.0, hyperparam_diversity / 2.0)
    weighted_diversity = (algo_diversity * 0.4 + 
                          norm_hyperparam * 0.4 +
                          min(1.0, perf_diversity) * 0.2)
    
    results['overall_diversity_index'] = weighted_diversity
    
    return results


def visualize_algorithm_distribution(clones: List, 
                                    output_path: Optional[str] = None,
                                    title: str = "Distribuição de Algoritmos",
                                    show_plot: bool = True) -> plt.Figure:
    """
    Visualiza a distribuição de algoritmos na população.
    
    Args:
        clones: Lista de clones
        output_path: Caminho para salvar a visualização (opcional)
        title: Título do gráfico
        show_plot: Se deve exibir o gráfico
        
    Returns:
        plt.Figure: Figura gerada
    """
    algorithms = [clone.algorithm_name for clone in clones]
    algo_counts = Counter(algorithms)
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Ordena por contagem
    labels, values = zip(*sorted(algo_counts.items(), key=lambda x: x[1], reverse=True))
    
    # Cria gráfico de barras
    ax.bar(labels, values, color=sns.color_palette("viridis", len(labels)))
    
    # Adiciona rótulos e título
    for i, v in enumerate(values):
        ax.text(i, v + 0.1, str(v), ha='center')
        
    ax.set_title(title)
    ax.set_ylabel("Contagem")
    ax.set_xlabel("Algoritmo")
    
    plt.tight_layout()
    
    # Salva a figura se um caminho for fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Exibe o gráfico
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fig


def visualize_diversity_metrics(diversity_history: List[Dict], 
                               output_path: Optional[str] = None,
                               title: str = "Evolução da Diversidade",
                               show_plot: bool = True) -> plt.Figure:
    """
    Visualiza a evolução de métricas de diversidade ao longo do tempo.
    
    Args:
        diversity_history: Lista de métricas de diversidade por geração
        output_path: Caminho para salvar a visualização (opcional)
        title: Título do gráfico
        show_plot: Se deve exibir o gráfico
        
    Returns:
        plt.Figure: Figura gerada
    """
    if not diversity_history:
        return None
        
    # Extrai dados para o gráfico
    generations = range(1, len(diversity_history) + 1)
    overall_diversity = [d['overall_diversity_index'] for d in diversity_history]
    algo_diversity = [d['algorithm_diversity']['simpson_index'] for d in diversity_history]
    perf_diversity = [d['performance_diversity']['score_cv'] 
                     if d['performance_diversity']['score_mean'] != 0 else 0 
                     for d in diversity_history]
    
    # Normaliza o coeficiente de variação de hiperparâmetros (pode ser maior que 1)
    hyperparam_diversity = [min(1.0, d['hyperparameter_diversity']['average_coefficient_of_variation'] / 2.0) 
                           for d in diversity_history]
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(generations, overall_diversity, 'o-', linewidth=2, label='Diversidade Geral', color='#1f77b4')
    ax.plot(generations, algo_diversity, 's-', linewidth=2, label='Diversidade de Algoritmos', color='#ff7f0e')
    ax.plot(generations, hyperparam_diversity, '^-', linewidth=2, label='Diversidade de Hiperparâmetros', color='#2ca02c')
    ax.plot(generations, perf_diversity, 'v-', linewidth=2, label='Diversidade de Desempenho', color='#d62728')
    
    # Configurações do gráfico
    ax.set_title(title)
    ax.set_xlabel('Geração')
    ax.set_ylabel('Índice de Diversidade')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Adiciona anotações para mudanças significativas
    for i in range(1, len(overall_diversity)):
        if abs(overall_diversity[i] - overall_diversity[i-1]) > 0.15:
            ax.annotate(f"{overall_diversity[i]:.2f}",
                       xy=(i+1, overall_diversity[i]),
                       xytext=(i+1, overall_diversity[i] + 0.05),
                       ha='center')
    
    plt.tight_layout()
    
    # Salva a figura se um caminho for fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Exibe o gráfico
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fig


def create_clone_embedding(clones: List,
                          method: str = 'tsne',
                          perplexity: int = 30) -> Tuple[np.ndarray, List]:
    """
    Cria uma representação de baixa dimensão dos clones para visualização.
    
    Args:
        clones: Lista de clones
        method: Método de redução de dimensionalidade ('tsne' ou 'pca')
        perplexity: Parâmetro para t-SNE
        
    Returns:
        Tuple: (matriz de coordenadas, lista de algoritmos)
    """
    # Extrai características dos clones
    features = []
    algorithms = []
    
    for clone in clones:
        # Características: vetor de características numéricas
        clone_vector = []
        
        # Algoritmo (usado para colorir os pontos)
        algorithms.append(clone.algorithm_name)
        
        # Adiciona hiperparâmetros numéricos
        param_values = {}
        
        # Verifica se o clone tem o atributo hyperparams
        if not hasattr(clone, 'hyperparams'):
            # Extrai parâmetros relevantes específicos de cada tipo de clone
            if hasattr(clone, 'algorithm_name'):
                param_values = {'algorithm': clone.algorithm_name}
                
                if hasattr(clone, 'n_clusters'):
                    param_values['n_clusters'] = clone.n_clusters
                if hasattr(clone, 'n_components'):
                    param_values['n_components'] = clone.n_components
                if hasattr(clone, 'seed'):
                    param_values['seed'] = clone.seed
        else:
            # Use hyperparams normalmente se estiver disponível
            param_values = clone.hyperparams
            
        # Processa os parâmetros para o vetor de características
        for param_name in ['alpha', 'C', 'max_iter', 'gamma', 'n_estimators', 'max_depth', 'min_samples_split', 'n_clusters', 'n_components', 'seed']:
            value = param_values.get(param_name, 0)
            if isinstance(value, (int, float)):
                clone_vector.append(float(value))
            else:
                clone_vector.append(0.0)
                
        # Adiciona tipo de vetorizador como característica binária, se existir
        if hasattr(clone, 'vectorizer_name'):
            clone_vector.append(1.0 if clone.vectorizer_name == 'tfidf' else 0.0)
        else:
            clone_vector.append(0.0)  # Valor padrão para clones sem vetorizador
        
        # Características de desempenho
        clone_vector.append(float(clone.performance_score) if clone.performance_score is not None else 0.0)
        
        # Geração
        clone_vector.append(float(clone.generation))
        
        features.append(clone_vector)
    
    # Verifica se temos dados suficientes
    if len(features) < 2:
        # Retorna um embedding simples para evitar erros
        dummy_embedding = np.array([[0, 0], [1, 1]])[:len(features)]
        return dummy_embedding, algorithms
        
    # Converte para array numpy
    X = np.array(features)
    
    # Normalização dos dados
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1  # Evita divisão por zero
    X_norm = (X - X_mean) / X_std
    
    # Aplica redução de dimensionalidade
    if method == 'tsne':
        # Ajusta a perplexidade com base no número de amostras
        # A perplexidade deve ser menor que o número de amostras
        adjusted_perplexity = min(perplexity, len(X_norm) - 1)
        if adjusted_perplexity < 1:
            adjusted_perplexity = 1
            
        # Verifica se temos dimensões suficientes
        if X_norm.shape[0] >= 3:  # t-SNE requer pelo menos 3 amostras
            embedding = TSNE(n_components=2, perplexity=adjusted_perplexity, random_state=42).fit_transform(X_norm)
        else:
            # Fallback para PCA ou simplesmente retorna os pontos
            if X_norm.shape[0] >= 2:  # PCA requer pelo menos 2 amostras
                embedding = PCA(n_components=2, random_state=42).fit_transform(X_norm)
            else:
                embedding = np.array([[0, 0]] * X_norm.shape[0])
    else:  # PCA
        if X_norm.shape[0] >= 2:  # PCA requer pelo menos 2 amostras
            embedding = PCA(n_components=2, random_state=42).fit_transform(X_norm)
        else:
            embedding = np.array([[0, 0]] * X_norm.shape[0])
        
    return embedding, algorithms


def visualize_clone_diversity(clones: List, 
                             method: str = 'tsne',
                             output_path: Optional[str] = None,
                             title: str = "Mapa de Diversidade de Clones",
                             show_plot: bool = True) -> plt.Figure:
    """
    Visualiza a diversidade de clones em um espaço bidimensional.
    
    Args:
        clones: Lista de clones
        method: Método de redução de dimensionalidade ('tsne' ou 'pca')
        output_path: Caminho para salvar a visualização (opcional)
        title: Título do gráfico
        show_plot: Se deve exibir o gráfico
        
    Returns:
        plt.Figure: Figura gerada
    """
    if len(clones) < 3:
        print(f"Aviso: Não há clones suficientes ({len(clones)}) para criar uma visualização de diversidade significativa.")
        if output_path:
            # Cria uma figura simples com uma mensagem
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Poucos clones para visualização\n(apenas {len(clones)} disponíveis)", 
                   ha='center', va='center', fontsize=14)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        return None
        
    # Cria embedding
    embedding, algorithms = create_clone_embedding(clones, method)
    
    # Cria DataFrame para Seaborn
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'algorithm': algorithms,
        'performance': [clone.performance_score if clone.performance_score is not None else 0 
                       for clone in clones],
        'generation': [clone.generation for clone in clones]
    })
    
    # Cria o gráfico
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Scatter plot com Seaborn
    scatter = sns.scatterplot(
        data=df,
        x='x', y='y',
        hue='algorithm',
        size='performance',
        sizes=(50, 200),
        palette='viridis',
        alpha=0.7,
        ax=ax
    )
    
    # Adiciona rótulos e título
    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} Dimensão 1")
    ax.set_ylabel(f"{method.upper()} Dimensão 2")
    
    # Remove os ticks dos eixos (não são significativos)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Adiciona anotações para alguns pontos representativos
    for i, clone in enumerate(clones):
        # Anota apenas alguns pontos para não sobrecarregar o gráfico
        if clone.performance_score and clone.performance_score >= 0.9:
            ax.annotate(f"{clone.id}",
                       xy=(embedding[i, 0], embedding[i, 1]),
                       xytext=(5, 5),
                       textcoords='offset points',
                       fontsize=8)
    
    # Melhora a legenda
    handles, labels = scatter.get_legend_handles_labels()
    ax.legend(handles, labels, title="Algoritmo", loc="upper right")
    
    plt.tight_layout()
    
    # Salva a figura se um caminho for fornecido
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Exibe o gráfico
    if show_plot:
        plt.show()
    else:
        plt.close()
        
    return fig 
