"""
Módulo de visualização para o sistema Mestre dos Clones.

Implementa funções para visualizar a evolução dos clones e suas relações de parentesco.
"""

import os
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from typing import Dict, List, Any, Optional


def load_evolution_history(filepath: str = "output/results/evolution_history.json") -> List[Dict]:
    """
    Carrega o histórico de evolução de um arquivo JSON.
    
    Args:
        filepath: Caminho para o arquivo JSON
        
    Returns:
        List[Dict]: Histórico de evolução
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        history = json.load(f)
    return history


def extract_genealogy_data(history: List[Dict]) -> Dict[str, Dict]:
    """
    Extrai dados genealógicos do histórico de evolução.
    
    Args:
        history: Histórico de evolução
        
    Returns:
        Dict: Informações sobre todos os clones e suas relações
    """
    clones_data = {}
    
    # Primeiro passo: reúne informações sobre todos os clones
    for gen_record in history:
        generation_num = gen_record['generation']
        
        for clone in gen_record['clones']:
            clone_id = clone['id']
            
            # Se já temos este clone registrado, apenas atualizamos informações
            if clone_id in clones_data:
                clones_data[clone_id]['last_seen_gen'] = generation_num
                clones_data[clone_id]['score'] = clone['score']
            else:
                # Caso contrário, registramos como um novo clone
                clones_data[clone_id] = {
                    'id': clone_id,
                    'name': clone['name'],
                    'algorithm': clone.get('algorithm', 'unknown'),
                    'vectorizer': clone.get('vectorizer', 'unknown'),
                    'parent_ids': clone.get('parent_ids', []),
                    'first_seen_gen': generation_num,
                    'last_seen_gen': generation_num,
                    'score': clone['score'],
                    'generation': clone.get('generation', 1)
                }
    
    return clones_data


def create_genealogy_graph(clones_data: Dict[str, Dict]) -> nx.DiGraph:
    """
    Cria um grafo dirigido representando a genealogia dos clones.
    
    Args:
        clones_data: Dados sobre todos os clones
        
    Returns:
        nx.DiGraph: Grafo de genealogia
    """
    G = nx.DiGraph()
    
    # Adiciona todos os clones como nós
    for clone_id, data in clones_data.items():
        G.add_node(
            clone_id,
            name=data['name'],
            algorithm=data['algorithm'],
            vectorizer=data['vectorizer'],
            first_seen=data['first_seen_gen'],
            last_seen=data['last_seen_gen'],
            score=data['score'],
            generation=data['generation']
        )
    
    # Adiciona arestas de parentesco
    for clone_id, data in clones_data.items():
        for parent_id in data['parent_ids']:
            if parent_id in clones_data:  # Garante que o pai existe no grafo
                G.add_edge(parent_id, clone_id)
    
    return G


def plot_genealogy_tree(G: nx.DiGraph, output_path: str = "output/visualizations/genealogy_tree.png"):
    """
    Visualiza a árvore genealógica dos clones.
    
    Args:
        G: Grafo de genealogia
        output_path: Caminho para salvar a visualização
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    plt.figure(figsize=(15, 10))
    
    # Usamos layout hierárquico para melhor visualização de parentesco
    pos = nx.spring_layout(G, seed=42)
    
    # Determinamos cores com base no algoritmo
    node_colors = []
    for node in G.nodes():
        algorithm = G.nodes[node]['algorithm']
        if algorithm == 'naive_bayes':
            color = 'skyblue'
        elif algorithm == 'logistic':
            color = 'lightgreen'
        elif algorithm == 'svm':
            color = 'salmon'
        else:
            color = 'lightgray'
        node_colors.append(color)
    
    # Determinamos tamanho dos nós com base na pontuação
    node_sizes = []
    for node in G.nodes():
        score = G.nodes[node]['score']
        if score is not None:
            node_sizes.append(300 + 700 * score)
        else:
            node_sizes.append(300)
    
    # Desenha o grafo
    nx.draw(
        G, pos,
        with_labels=False,
        node_color=node_colors,
        node_size=node_sizes,
        font_size=8,
        edge_color='gray',
        arrowsize=10
    )
    
    # Adiciona rótulos apenas para clones com pontuação alta
    labels = {}
    for node in G.nodes():
        score = G.nodes[node]['score']
        name = G.nodes[node]['name']
        if score is not None and score > 0.7:  # Apenas clones com pontuação alta
            labels[node] = f"{name}\n({score:.2f})"
    
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
    
    plt.title("Árvore Genealógica dos Clones", fontsize=16)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def visualize_clone_lineage(
        history_filepath: str = "output/results/evolution_history.json",
        output_path: str = "output/visualizations/genealogy_tree.png") -> str:
    """
    Função principal para visualizar a linhagem dos clones.
    
    Args:
        history_filepath: Caminho para o arquivo de histórico
        output_path: Caminho para salvar a visualização
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    # Carrega histórico
    history = load_evolution_history(history_filepath)
    
    # Extrai dados de genealogia
    clones_data = extract_genealogy_data(history)
    
    # Cria grafo
    G = create_genealogy_graph(clones_data)
    
    # Visualiza e salva
    return plot_genealogy_tree(G, output_path)


def plot_algorithm_distribution(
        history: List[Dict],
        output_path: str = "output/visualizations/algorithm_distribution.png"):
    """
    Visualiza a distribuição de algoritmos ao longo das gerações.
    
    Args:
        history: Histórico de evolução
        output_path: Caminho para salvar a visualização
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    # Extrai contagem de algoritmos por geração
    generations = []
    naive_bayes_counts = []
    logistic_counts = []
    svm_counts = []
    other_counts = []
    
    for gen_record in history:
        gen_num = gen_record['generation']
        generations.append(gen_num)
        
        # Inicializa contagens
        nb_count = 0
        lr_count = 0
        svm_count = 0
        other_count = 0
        
        # Conta algoritmos nesta geração
        for clone in gen_record['clones']:
            algo = clone.get('algorithm', 'unknown')
            if algo == 'naive_bayes':
                nb_count += 1
            elif algo == 'logistic':
                lr_count += 1
            elif algo == 'svm':
                svm_count += 1
            else:
                other_count += 1
        
        naive_bayes_counts.append(nb_count)
        logistic_counts.append(lr_count)
        svm_counts.append(svm_count)
        other_counts.append(other_count)
    
    # Cria o gráfico
    plt.figure(figsize=(12, 6))
    
    plt.stackplot(
        generations,
        naive_bayes_counts, logistic_counts, svm_counts, other_counts,
        labels=['Naive Bayes', 'Regressão Logística', 'SVM', 'Outros'],
        alpha=0.8,
        colors=['skyblue', 'lightgreen', 'salmon', 'lightgray']
    )
    
    plt.xlabel('Geração')
    plt.ylabel('Quantidade de Clones')
    plt.title('Distribuição de Algoritmos por Geração')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path


def plot_evolution_progress(
        history: List[Dict], 
        output_path: str = "output/visualizations/evolution_progress.png",
        show_plot: bool = False) -> str:
    """
    Gera gráfico mostrando o progresso da evolução ao longo das gerações.
    
    Args:
        history: Histórico de evolução
        output_path: Caminho para salvar a visualização
        show_plot: Se deve mostrar o gráfico durante a execução
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    # Extrai dados do histórico
    generations = []
    best_scores = []
    avg_scores = []
    
    for gen_record in history:
        generations.append(gen_record['generation'])
        
        # Extrai a melhor pontuação da geração
        best_score = gen_record.get('best_score')
        if best_score is None and 'clones' in gen_record:
            # Se não tiver best_score, calcula a partir dos clones
            clone_scores = [c['score'] for c in gen_record['clones'] if c['score'] is not None]
            best_score = max(clone_scores) if clone_scores else None
        best_scores.append(best_score if best_score is not None else 0)
        
        # Extrai a pontuação média da geração
        avg_score = gen_record.get('avg_score')
        if avg_score is None and 'clones' in gen_record:
            # Se não tiver avg_score, calcula a partir dos clones
            clone_scores = [c['score'] for c in gen_record['clones'] if c['score'] is not None]
            avg_score = sum(clone_scores) / len(clone_scores) if clone_scores else None
        avg_scores.append(avg_score if avg_score is not None else 0)
    
    # Cria o gráfico
    plt.figure(figsize=(12, 6))
    
    plt.plot(generations, best_scores, 'o-', linewidth=2, label='Melhor Clone', color='#1f77b4')
    plt.plot(generations, avg_scores, 's-', linewidth=2, label='Média da População', color='#ff7f0e')
    
    plt.xlabel('Geração')
    plt.ylabel('Pontuação (F1)')
    plt.title('Evolução do Desempenho dos Clones')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Adiciona anotações para os melhores pontos
    for i, score in enumerate(best_scores):
        if i == 0 or score >= max(best_scores[:i]):
            plt.annotate(
                f"{score:.3f}",
                xy=(generations[i], score),
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # Configurações finais
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path


def plot_generation_metrics(
        generation_record: Dict,
        output_path: str = "output/visualizations/generation_metrics.png",
        show_plot: bool = False) -> str:
    """
    Visualiza métricas detalhadas de uma geração específica.
    
    Args:
        generation_record: Registro de uma geração específica
        output_path: Caminho para salvar a visualização
        show_plot: Se deve mostrar o gráfico durante a execução
        
    Returns:
        str: Caminho para o arquivo salvo
    """
    # Extrai dados dos clones
    clone_names = []
    clone_scores = []
    clone_algos = []
    
    for clone in generation_record['clones']:
        # Usa apenas os primeiros 10 caracteres do nome para legibilidade
        clone_names.append(clone['name'][:10] + "...")
        clone_scores.append(clone['score'] if clone['score'] is not None else 0)
        clone_algos.append(clone['algorithm'])
    
    # Ordena os clones por pontuação
    sorted_indices = np.argsort(clone_scores)[::-1]  # Ordem decrescente
    clone_names = [clone_names[i] for i in sorted_indices]
    clone_scores = [clone_scores[i] for i in sorted_indices]
    clone_algos = [clone_algos[i] for i in sorted_indices]
    
    # Define cores por algoritmo
    colors = []
    for algo in clone_algos:
        if algo == 'naive_bayes':
            colors.append('skyblue')
        elif algo == 'logistic':
            colors.append('lightgreen')
        elif algo == 'svm':
            colors.append('salmon')
        else:
            colors.append('lightgray')
    
    # Cria o gráfico
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(clone_names, clone_scores, color=colors, alpha=0.8)
    
    # Adiciona rótulos e títulos
    plt.xlabel('Clone')
    plt.ylabel('Pontuação (F1)')
    plt.title(f'Métricas da Geração {generation_record["generation"]}')
    
    # Adiciona a pontuação em cima de cada barra
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.01,
            f'{height:.3f}',
            ha='center',
            va='bottom',
            rotation=90,
            fontsize=8
        )
    
    # Adiciona legenda para cores
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', label='Naive Bayes'),
        Patch(facecolor='lightgreen', label='Regressão Logística'),
        Patch(facecolor='salmon', label='SVM'),
        Patch(facecolor='lightgray', label='Outros')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Rotaciona os rótulos do eixo x para melhor legibilidade
    plt.xticks(rotation=45, ha='right')
    
    # Configurações finais
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return output_path
