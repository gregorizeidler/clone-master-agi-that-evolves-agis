#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Advanced visualizations for the Clone Master system.
This module provides interactive visualizations without affecting core functionality.
"""

import os
import json
import pickle
import numpy as np
import plotly.graph_objects as go
import networkx as nx
import streamlit as st

def load_evolution_history():
    """
    Loads evolution history from JSON file.
    
    Returns:
        dict: Evolution history data or None if file doesn't exist
    """
    history_path = "output/results/evolution_history.json"
    if not os.path.exists(history_path):
        return None
    
    try:
        with open(history_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading evolution history: {str(e)}")
        return None

def load_clones(generation=None):
    """
    Loads clones from pickle files.
    
    Args:
        generation (int, optional): Specific generation to load. If None, loads all.
    
    Returns:
        list: List of loaded clone objects
    """
    clones = []
    base_dir = "output/clones"
    
    # Get all generation directories
    generation_dirs = []
    if generation is not None:
        gen_dir = f"generation_{generation}"
        if os.path.exists(os.path.join(base_dir, gen_dir)):
            generation_dirs = [gen_dir]
    else:
        if os.path.exists(base_dir):
            generation_dirs = [d for d in os.listdir(base_dir) 
                             if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('generation_')]
    
    # Load clones from each generation
    for gen_dir in generation_dirs:
        gen_path = os.path.join(base_dir, gen_dir)
        
        # Get directories within the generation directory
        clone_dirs = [d for d in os.listdir(gen_path) 
                     if os.path.isdir(os.path.join(gen_path, d))]
        
        for clone_dir in clone_dirs:
            # Try to find the clone file in the directory
            clone_path = os.path.join(gen_path, clone_dir, clone_dir)
            
            # If there's no direct match, look for any .pkl file in the directory
            if not os.path.exists(clone_path) and not os.path.exists(clone_path + ".pkl"):
                pkl_files = [f for f in os.listdir(os.path.join(gen_path, clone_dir)) 
                            if f.endswith('.pkl')]
                if pkl_files:
                    clone_path = os.path.join(gen_path, clone_dir, pkl_files[0])
            else:
                # Try with .pkl extension
                if os.path.exists(clone_path + ".pkl"):
                    clone_path += ".pkl"
            
            try:
                with open(clone_path, 'rb') as f:
                    clone = pickle.load(f)
                    clones.append(clone)
            except Exception as e:
                # Just continue if we couldn't load this clone
                # print(f"Error loading {clone_dir}: {str(e)}")
                continue
    
    return clones

def create_interactive_evolution_chart():
    """
    Creates an interactive chart showing the evolution of performance and diversity.
    
    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure
    """
    # Load evolution history
    history = load_evolution_history()
    if not history:
        st.warning("No evolution history found. Run the evolution process first.")
        return None
    
    # Check if we have generations data in the expected format
    if 'generations' in history:
        generations_data = history['generations']
    else:
        # If the data is directly a list of generations
        generations_data = history
    
    if not generations_data or not isinstance(generations_data, list):
        st.warning("Invalid evolution history format. Please run the evolution process again.")
        return None
    
    # Extract data
    generations = list(range(1, len(generations_data) + 1))
    best_scores = []
    avg_scores = []
    
    # Handle different possible data formats
    for gen in generations_data:
        if isinstance(gen, dict):
            # Extract scores using various possible keys
            best_score = gen.get('best_score', None)
            if best_score is None and 'best_clone' in gen and isinstance(gen['best_clone'], dict):
                best_score = gen['best_clone'].get('score', 0)
            
            avg_score = gen.get('avg_score', 0)
            
            best_scores.append(best_score if best_score is not None else 0)
            avg_scores.append(avg_score if avg_score is not None else 0)
        else:
            # Skip invalid entries
            st.warning(f"Invalid generation data format: {type(gen)}")
    
    # Check if we have valid data
    if not best_scores or all(score == 0 for score in best_scores):
        st.warning("No valid performance data found in the evolution history.")
        return None
    
    # Extract diversity if available
    diversity_data = []
    algorithm_diversity = []
    
    for gen in generations_data:
        if isinstance(gen, dict) and 'diversity_metrics' in gen:
            diversity_metrics = gen['diversity_metrics']
            if isinstance(diversity_metrics, dict):
                # Ensure we get numeric values
                overall_div = diversity_metrics.get('overall_diversity', 0)
                algo_div = diversity_metrics.get('algorithm_diversity', 0)
                
                # Convert to float if possible, otherwise use 0
                try:
                    overall_div = float(overall_div) if overall_div is not None else 0
                except (ValueError, TypeError):
                    overall_div = 0
                
                try:
                    algo_div = float(algo_div) if algo_div is not None else 0
                except (ValueError, TypeError):
                    algo_div = 0
                
                diversity_data.append(overall_div)
                algorithm_diversity.append(algo_div)
            else:
                diversity_data.append(0)
                algorithm_diversity.append(0)
        else:
            diversity_data.append(0)
            algorithm_diversity.append(0)
    
    # Create interactive plot
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=generations, 
        y=best_scores, 
        mode='lines+markers',
        name='Best Score',
        line=dict(color='#00CC96', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=avg_scores, 
        mode='lines+markers',
        name='Average Score',
        line=dict(color='#636EFA', width=2, dash='dot')
    ))
    
    # Only add diversity traces if we have meaningful data
    if any(isinstance(d, (int, float)) and d > 0 for d in diversity_data):
        fig.add_trace(go.Scatter(
            x=generations, 
            y=diversity_data, 
            mode='lines+markers',
            name='Overall Diversity',
            line=dict(color='#FFA15A', width=2),
            visible='legendonly'  # Hide by default, toggle from legend
        ))
    
    if any(isinstance(d, (int, float)) and d > 0 for d in algorithm_diversity):
        fig.add_trace(go.Scatter(
            x=generations, 
            y=algorithm_diversity, 
            mode='lines+markers',
            name='Algorithm Diversity',
            line=dict(color='#FF6692', width=2),
            visible='legendonly'  # Hide by default, toggle from legend
        ))
    
    # Add range slider
    fig.update_layout(
        title='Interactive Evolution Progress',
        xaxis=dict(
            title='Generation',
            tickmode='linear',
            range=[1, len(generations)],
            rangeslider=dict(visible=True),
            type='linear'
        ),
        yaxis=dict(
            title='Score / Diversity',
            range=[0, 1.05],
        ),
        hovermode='x unified',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    # Add annotations for best clone in each generation
    annotations = []
    for i, gen in enumerate(generations_data):
        if isinstance(gen, dict) and 'best_clone' in gen:
            best_clone = gen['best_clone']
            clone_name = None
            
            if isinstance(best_clone, dict):
                clone_name = best_clone.get('name', None)
            
            # If we have a valid name and score, add annotation
            if clone_name and i < len(best_scores) and best_scores[i] > 0:
                annotations.append(dict(
                    x=i+1,
                    y=best_scores[i],
                    text=f"{clone_name}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="#636363",
                    ax=0,
                    ay=-30,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=4,
                    bgcolor="#ffffff",
                    opacity=0.8
                ))
    
    # Only add a subset of annotations to avoid crowding (every 2 generations or first/last)
    if annotations:
        selected_annotations = [
            ann for i, ann in enumerate(annotations) 
            if i == 0 or i == len(annotations)-1 or i % 2 == 0
        ]
        fig.update_layout(annotations=selected_annotations)
    
    return fig

def create_genealogy_graph():
    """
    Creates an interactive genealogy graph of all clones.
    
    Returns:
        plotly.graph_objects.Figure: Interactive plotly figure with genealogy graph
    """
    # Load all clones
    clones = load_clones()
    if not clones:
        st.warning("No clones found. Run the evolution process first.")
        return None
    
    st.info(f"Found {len(clones)} clones to display in the genealogy tree.")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add nodes (clones)
    for clone in clones:
        # Extract key attributes for display
        clone_id = getattr(clone, 'id', 'unknown')
        generation = getattr(clone, 'generation', 0)
        name = getattr(clone, 'name', f'Clone-{clone_id}')
        score = getattr(clone, 'performance_score', 0)
        
        # Try different attribute names for algorithm
        algorithm = 'unknown'
        for attr in ['algorithm_name', 'algorithm', 'algo_name']:
            if hasattr(clone, attr):
                algorithm = getattr(clone, attr)
                break
        
        clone_type = clone.__class__.__name__
        
        # Only add if we have a valid ID
        if clone_id and clone_id != 'unknown':
            # Store as node
            G.add_node(
                clone_id,
                name=name,
                generation=generation,
                score=score if score is not None else 0,
                algorithm=algorithm,
                type=clone_type
            )
    
    # Add edges (parent-child relationships)
    for clone in clones:
        child_id = getattr(clone, 'id', 'unknown')
        # Check different attribute names for parents
        parent_ids = []
        for attr in ['parent_ids', 'parents', 'parent_id']:
            if hasattr(clone, attr):
                p_ids = getattr(clone, attr)
                if isinstance(p_ids, list):
                    parent_ids = p_ids
                elif p_ids:  # Single parent ID
                    parent_ids = [p_ids]
                break
        
        if child_id and child_id != 'unknown' and child_id in G:
            for parent_id in parent_ids:
                if parent_id and parent_id in G:  # Only add if parent exists
                    G.add_edge(parent_id, child_id)
    
    # Check if we have a valid graph
    if not G.nodes or len(G.nodes) < 2:
        st.warning("Not enough valid clones to create a genealogy graph.")
        return None
    
    # Use layout appropriate for DAGs (directed acyclic graphs)
    try:
        # Try using dot layout if available (requires pygraphviz)
        try:
            pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        except:
            # If pygraphviz is not available, use hierarchical layout
            pos = nx.multipartite_layout(G, subset_key="generation")
    except Exception as e:
        # Fall back to a simple layout if all else fails
        st.warning(f"Using simple layout due to error: {str(e)}")
        pos = nx.spring_layout(G, seed=42)
    
    # Create node groups by generation for coloring
    node_generations = [G.nodes[node]['generation'] for node in G.nodes]
    max_gen = max(node_generations) if node_generations else 0
    
    # Create node text for hover information
    hover_texts = []
    for node in G.nodes:
        node_data = G.nodes[node]
        text = (
            f"ID: {node}<br>"
            f"Name: {node_data['name']}<br>"
            f"Generation: {node_data['generation']}<br>"
            f"Type: {node_data['type']}<br>"
            f"Algorithm: {node_data['algorithm']}<br>"
            f"Score: {node_data['score']:.4f}"
        )
        hover_texts.append(text)
    
    # Calculate node sizes based on score (better scores are larger)
    node_sizes = []
    for node in G.nodes:
        score = G.nodes[node]['score']
        if score is None:
            score = 0
        # Make size between 10 and 30 based on score
        size = 10 + (score * 20) if score else 10
        node_sizes.append(size)
    
    # Create edge trace
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        # Add None to create separation between edges
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        name='Parent-Child Relationship'
    )
    
    # Create node trace
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=hover_texts,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            color=node_generations,
            size=node_sizes,
            colorbar=dict(
                title='Generation',
                thickness=15,
                tickvals=list(range(0, max_gen+1, max(1, max_gen//5))),
                ticktext=list(range(0, max_gen+1, max(1, max_gen//5)))
            ),
            line=dict(width=2)
        ),
        name='Clones'
    )
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title=dict(
                          text='Interactive Clone Genealogy Tree',
                          font=dict(size=16)
                      ),
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20, l=5, r=5, t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      annotations=[
                          dict(
                              text="Larger nodes represent better performance. Color indicates generation.",
                              showarrow=False,
                              xref="paper", 
                              yref="paper",
                              x=0.01, 
                              y=-0.01,
                              font=dict(size=12)
                          )
                      ]
                  ))
    
    # Add zoom and pan tools
    fig.update_layout(
        updatemenus=[
            dict(
                type = "buttons",
                direction = "left",
                buttons=[
                    dict(
                        args=[{"visible": [True, True]}],
                        label="Show All",
                        method="update"
                    ),
                    dict(
                        args=[{"visible": [False, True]}],
                        label="Hide Edges",
                        method="update"
                    ),
                ],
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.11,
                xanchor="left",
                y=1.1,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def render_features_page():
    """Renders the Features page in the Streamlit application."""
    st.title("🔬 Advanced Visualizations")
    st.write("""
    This page showcases interactive visualizations of the Clone Master system.
    These features are experimental and don't affect the core functionality.
    """)
    
    # Create tabs for different visualizations
    tab1, tab2 = st.tabs(["Interactive Evolution Chart", "Genealogy Tree"])
    
    with tab1:
        st.subheader("Evolution Progress")
        st.write("""
        This interactive chart shows how performance and diversity metrics change across generations.
        You can zoom, pan, and toggle different metrics using the legend.
        """)
        
        fig = create_interactive_evolution_chart()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Add extra information
            st.info("""
            **Tips:**
            - Click on legend items to show/hide metrics
            - Use the rangeslider at the bottom to select specific generations
            - Hover over points to see detailed metrics
            - Click and drag to zoom into specific areas
            """)
    
    with tab2:
        st.subheader("Clone Genealogy")
        st.write("""
        This visualization shows the family tree of all clones, tracing their parentage across generations.
        Hover over nodes to see details about each clone.
        """)
        
        # Add filters for the genealogy tree
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Filters coming soon:**")
            st.checkbox("Show only best performers", value=False, disabled=True)
        
        with col2:
            st.write("**Display options:**")
            st.checkbox("Highlight islands", value=False, disabled=True)
        
        # Display the genealogy tree
        fig = create_genealogy_graph()
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
            # Add extra information
            st.info("""
            **Interaction guide:**
            - Hover over nodes to see clone details
            - Zoom in/out with scroll wheel or pinch gesture
            - Pan by clicking and dragging
            - Use the buttons to show/hide connections
            """)
        
        st.warning("""
        Note: For very large genealogies (hundreds of clones), this visualization may become slow.
        Future updates will optimize performance for larger evolutionary runs.
        """)

if __name__ == "__main__":
    # For testing the module independently
    st.set_page_config(
        page_title="Clone Master - Advanced Features",
        page_icon="🧬",
        layout="wide"
    )
    render_features_page() 
