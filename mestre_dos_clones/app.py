#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Web interface for the Clone Master using Streamlit.
Allows visualizing, interacting with, and testing clones in a user-friendly way.
"""

import os
import time
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# Import the features module
from features import render_features_page

# Page settings
st.set_page_config(
    page_title="Clone Master - AGI that Creates AGIs",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🧬 Clone Master")
st.subheader("Self-evolving multi-agent AGI system")

# Sidebar for navigation
st.sidebar.title("Navigation")
pagina = st.sidebar.radio(
    "Select an option:",
    ["Home", "Run Evolution", "View Results", "Test Clones", "Features", "Settings", "About"]
)

# Function to check if the system has been run
def sistema_executado():
    """Checks if there are results from previous runs."""
    return os.path.exists("output/results/evolution_history.json")

# Function to load evolution history
def carregar_historico():
    """Loads evolution history from the JSON file."""
    try:
        with open("output/results/evolution_history.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading history: {str(e)}")
        return None

# Function to list available generations
def listar_geracoes():
    """Lists all available generations in the output directory."""
    if not os.path.exists("output/clones"):
        return []
    
    geracoes = []
    for item in os.listdir("output/clones"):
        if item.startswith("generation_"):
            try:
                gen_num = int(item.split("_")[1])
                geracoes.append(gen_num)
            except:
                pass
    
    return sorted(geracoes)

# Function to list clones of a generation
def listar_clones(geracao):
    """Lists all clones of a specific generation."""
    diretorio = f"output/clones/generation_{geracao}"
    if not os.path.exists(diretorio):
        return []
    
    clones = []
    for item in os.listdir(diretorio):
        # Checks if it's a directory containing a file with the same name
        if os.path.isdir(os.path.join(diretorio, item)):
            # Confirms if the file exists inside the directory
            if os.path.isfile(os.path.join(diretorio, item, item)):
                clones.append(item)
        # Or if it's directly a .pkl file
        elif item.endswith(".pkl") and os.path.isfile(os.path.join(diretorio, item)):
            clones.append(item)
    
    return sorted(clones)

# Function to load a clone
def carregar_clone(geracao, nome_clone):
    """Loads a specific clone from its file."""
    # First checks as a directory containing the file
    diretorio_clone = f"output/clones/generation_{geracao}/{nome_clone}"
    caminho_arquivo_dentro_dir = f"{diretorio_clone}/{nome_clone}"
    
    # Alternative path as a direct file
    caminho_arquivo_direto = f"output/clones/generation_{geracao}/{nome_clone}"
    
    # First tries the path as a directory
    if os.path.isdir(diretorio_clone) and os.path.isfile(caminho_arquivo_dentro_dir):
        caminho = caminho_arquivo_dentro_dir
    else:
        # Otherwise, tries as a direct file
        caminho = caminho_arquivo_direto
    
    try:
        with open(caminho, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading clone: {str(e)}")
        st.write(f"Path tried: {caminho}")
        return None

# Function to classify text using a clone
def classificar_texto(clone, texto):
    """Classifies a text using the provided clone."""
    try:
        predicao = clone.predict([texto])[0]
        sentimento = "positive" if predicao == 1 else "negative"
        return sentimento, predicao
    except Exception as e:
        st.error(f"Error classifying text: {str(e)}")
        return None, None

# === Home Page ===
if pagina == "Home":
    st.markdown("""
    ## Welcome to Clone Master!
    
    This system creates, trains, evaluates, and evolves specialized artificial intelligences **without direct human intervention**.
    
    Inspired by concepts of natural evolution and genetic engineering, the project uses autonomous agents
    that work in cycles to generate "clones" increasingly capable of solving specific tasks.
    
    ### 🧠 Agent Architecture
    
    The system consists of four main agents:
    
    - **🏗️ Intelligence Architect**: Designs new clones, defining their structure and logic
    - **🎓 Skills Trainer**: Trains each clone with data to perform specific tasks
    - **📊 Performance Evaluator**: Measures the performance of each clone in controlled scenarios
    - **🧬 Evolutionary Selector**: Decides which clones survive, evolve, or are discarded
    
    ### 🚀 Getting Started
    
    Go to the **"Run Evolution"** section to start the evolutionary process, or navigate to other
    sections to view results from previous runs and test the generated clones.
    """)
    
    # Checks if the system has already been run
    if sistema_executado():
        st.success("✅ The system has already been run. Explore other sections to see the results.")
    else:
        st.warning("⚠️ The system has not been run yet. Go to 'Run Evolution' to get started.")

    # Visual tutorial
    st.subheader("📚 Quick Tutorial")
    
    with st.expander("How to use this interface?", expanded=True):
        st.markdown("""
        Here's a quick guide to using Clone Master:
        
        1️⃣ **Run Evolution** - Configure and start the evolutionary process
        2️⃣ **View Results** - Explore statistics and performance graphs
        3️⃣ **Test Clones** - Try the clones with your own texts
        4️⃣ **About** - Learn more about the project and its architecture
        
        The recommended flow is to follow the order above, but you can navigate freely.
        """)
        
        # Simple diagram using ASCII characters
        st.code("""
        ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
        │  Architect  │────>│   Trainer   │────>│  Evaluator  │────>│  Selector   │
        └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                │                                                          │
                └──────────────────────────────────────────────────────────┘
                                          │
                                          v
                                  New Clone Generations
        """, language=None)
        
        st.info("💡 Tip: If you're just starting, begin with a small population (10 clones) and few generations (5) to see results more quickly.")

# === Run Evolution Page ===
elif pagina == "Run Evolution":
    st.header("Run AGI Evolution")
    
    st.markdown("""
    In this section, you can start the evolutionary process of AGIs.
    Select the desired task and configure the parameters to begin.
    """)
    
    # Task type selector (updated to show only fully implemented tasks)
    tipo_tarefa = st.selectbox(
        "Select task type:",
        [
            "Sentiment Classification (default)",
            "Numerical Regression (value prediction)",
            "Clustering",
            "Dimensionality Reduction"
        ],
        index=0,
        help="Select the type of task the clones should learn to solve"
    )
    
    # Exibe descrição e requisitos da tarefa selecionada
    if tipo_tarefa == "Sentiment Classification (default)":
        st.info("The clones will be evolved to classify texts as positive or negative. Requires labeled text data.")
        
    elif tipo_tarefa == "Numerical Regression (value prediction)":
        st.info("The clones will be evolved to predict continuous values. Ideal for price prediction, numerical estimations, and other regression tasks.")
        
    elif tipo_tarefa == "Clustering":
        st.info("The clones will be specialized in grouping similar data without predefined labels. Identifies natural patterns in data.")
        
    elif tipo_tarefa == "Dimensionality Reduction":
        st.info("The clones will be evolved to reduce the dimensionality of data, facilitating visualization and improving performance of subsequent models.")
    
    # Formulário para configurar a execução
    with st.form("form_execucao"):
        col1, col2 = st.columns(2)
        
        with col1:
            num_geracoes = st.slider(
                "Number of generations:",
                min_value=1,
                max_value=50,
                value=10,
                help="How many generations of clones will be evolved"
            )
            
            tamanho_populacao = st.slider(
                "Population size:",
                min_value=5,
                max_value=50,
                value=10,
                help="How many clones will exist in each generation"
            )
        
        with col2:
            # Parâmetros específicos para cada tipo de tarefa
            if tipo_tarefa == "Sentiment Classification (default)":
                usar_dados_exemplo = st.checkbox(
                    "Use example data",
                    value=True,
                    help="Use the example dataset included in the project"
                )
                
                arquivo_dados = None
                if not usar_dados_exemplo:
                    arquivo_dados = st.file_uploader(
                        "Upload your own data file:",
                        type=["txt", "csv"],
                        help="File format: 'text\\tlabel' (0=negative, 1=positive)"
                    )
            
            elif tipo_tarefa == "Numerical Regression (value prediction)":
                usar_dados_exemplo = st.checkbox(
                    "Use synthetic data",
                    value=True,
                    help="Use synthetic data generated automatically"
                )
                
                arquivo_dados = None
                if not usar_dados_exemplo:
                    arquivo_dados = st.file_uploader(
                        "Upload your own data file:",
                        type=["csv"],
                        help="CSV file with features in columns and target in the last column"
                    )
                
                metrica_erro = st.selectbox(
                    "Error Metric:",
                    ["MSE (Mean Squared Error)", "MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)"],
                    index=0,
                    help="Metric used to evaluate the performance of regression models"
                )
            
            elif tipo_tarefa == "Clustering":
                usar_dados_exemplo = st.checkbox(
                    "Use synthetic data",
                    value=True,
                    help="Use synthetic data generated automatically with well-defined clusters"
                )
                
                arquivo_dados = None
                if not usar_dados_exemplo:
                    arquivo_dados = st.file_uploader(
                        "Upload your own data file:",
                        type=["csv"],
                        help="CSV file with features for clustering"
                    )
                
                num_clusters = st.slider(
                    "Number of clusters:",
                    min_value=2,
                    max_value=10,
                    value=3,
                    help="Number of groups to be identified in the data"
                )
            
            elif tipo_tarefa == "Dimensionality Reduction":
                usar_dados_exemplo = st.checkbox(
                    "Use synthetic data",
                    value=True,
                    help="Use synthetic data of high dimensionality generated automatically"
                )
                
                arquivo_dados = None
                if not usar_dados_exemplo:
                    arquivo_dados = st.file_uploader(
                        "Upload your own data file:",
                        type=["csv"],
                        help="CSV file with high dimensionality features"
                    )
                
                num_componentes = st.slider(
                    "Number of components:",
                    min_value=2,
                    max_value=10,
                    value=2,
                    help="Number of dimensions to reduce the data"
                )
            
            # Configurações comuns para todas as tarefas
            semente = st.number_input(
                "Random seed:",
                min_value=1,
                max_value=10000,
                value=42,
                help="Seed for reproducibility of results"
            )
            
            usar_ilhas = st.checkbox(
                "Use evolutionary islands system",
                value=True,
                help="Divide the population into isolated islands with occasional migrations, increasing genetic diversity"
            )
        
        # Botão para iniciar evolução
        executar = st.form_submit_button("Start Evolution")
    
    # Lógica para executar o sistema
    if executar:
        # Importa o módulo para executar a evolução
        try:
            import sys
            import subprocess
            from main import run_evolution_cycle
            
            # Cria um spinner durante a execução
            with st.spinner(f"Executing evolutionary cycle for {tipo_tarefa}... This process can take several minutes depending on the parameters chosen."):
                st.info(f"Starting evolution with {num_geracoes} generations and population of {tamanho_populacao} clones.")
                
                # Executa diretamente a função principal
                try:
                    # Configura caminho do arquivo de dados se fornecido
                    data_filepath = None
                    if not usar_dados_exemplo and arquivo_dados:
                        # Salva o arquivo temporariamente
                        temp_file = os.path.join("temp_data", arquivo_dados.name)
                        os.makedirs("temp_data", exist_ok=True)
                        
                        with open(temp_file, "wb") as f:
                            f.write(arquivo_dados.getbuffer())
                        
                        data_filepath = temp_file
                    
                    # Prepara parâmetros específicos da tarefa
                    task_params = {
                        "data_filepath": data_filepath
                    }
                    
                    # Mapeia o tipo de tarefa para o parâmetro correto
                    if tipo_tarefa == "Sentiment Classification (default)":
                        task_params["task_type"] = "sentiment"
                    elif tipo_tarefa == "Numerical Regression (value prediction)":
                        task_params["task_type"] = "regressão"
                        task_params["error_metric"] = metrica_erro.split(" ")[0].lower()  # Extrai o primeiro termo (MSE, MAE, RMSE)
                    elif tipo_tarefa == "Clustering":
                        task_params["task_type"] = "clustering"
                        task_params["n_clusters"] = num_clusters
                    elif tipo_tarefa == "Dimensionality Reduction":
                        task_params["task_type"] = "dimensão"
                        task_params["n_components"] = num_componentes
                    
                    # Executa o ciclo evolutivo com os parâmetros definidos
                    run_evolution_cycle(
                        num_generations=num_geracoes,
                        population_size=tamanho_populacao,
                        seed=semente,
                        use_islands=usar_ilhas,
                        advanced_clones=True,
                        enable_diversity_tracking=True,
                        **task_params
                    )
                    
                    # Gera a visualização da árvore genealógica
                    from utils.visualization import visualize_clone_lineage
                    visualize_clone_lineage()
                    
                    st.success(f"✅ Evolution completed successfully! {num_geracoes} generations were evolved for the {tipo_tarefa} task.")
                    st.info("Go to the 'View Results' section to see the generated graphs and metrics.")
                    
                    # Adiciona um botão para ir diretamente para a visualização
                    if st.button("View Results Now"):
                        st.experimental_set_query_params(pagina="View Results")
                        st.experimental_rerun()
                    
                except Exception as e:
                    st.error(f"Error during execution: {str(e)}")
                    st.code(str(e), language="python")
                    
                    # Mostra informações adicionais de depuração
                    import traceback
                    st.expander("Error Details", expanded=False).code(traceback.format_exc())
                    
                    # Sugestão para implementação futura
                    st.warning("""
                    Note: This feature is under development. Currently, only the 'Sentiment Classification' task is fully implemented.
                    
                    Other selectable tasks in this interface will be implemented in future versions.
                    """)
        
        except ImportError as e:
            st.error(f"Error importing required modules: {str(e)}")
            
            # Oferece alternativa via linha de comando
            st.warning("You can still execute the process via command line:")
            
            # Cria um comando que seria executado
            comando = f"cd /Users/gregorizeidler/test\\ agi\\ 2/mestre_dos_clones && python -m mestre_dos_clones.main --generations {num_geracoes} --population {tamanho_populacao}"
            
            if usar_ilhas:
                comando += " --islands"
                
            comando += f" --seed {semente}"
            
            st.code(comando, language="bash")

# === View Results Page ===
elif pagina == "View Results":
    st.header("View Results")
    
    # Verifica se o sistema já foi executado
    if not sistema_executado():
        st.warning("⚠️ The system has not been run yet. Go to 'Run Evolution' to get started.")
        st.stop()
    
    # Carrega o histórico de evolução
    historico = carregar_historico()
    if historico is None:
        st.error("Unable to load evolution history.")
        st.stop()
    
    # Mostra estatísticas gerais
    st.subheader("General Statistics")
    
    total_geracoes = len(historico)
    ultima_geracao = historico[-1]
    melhor_score_geral = max([gen.get('best_score', 0) for gen in historico])
    melhor_geracao = next((i+1 for i, gen in enumerate(historico) if gen.get('best_score', 0) == melhor_score_geral), None)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total of Generations", total_geracoes)
    with col2:
        st.metric("Best Score", f"{melhor_score_geral:.4f}")
    with col3:
        st.metric("Best Generation", melhor_geracao)
    
    # Gráfico de evolução de desempenho
    st.subheader("Evolution of Performance")
    
    if os.path.exists("output/visualizations/evolution_progress.png"):
        st.image("output/visualizations/evolution_progress.png")
    else:
        # Cria o gráfico se o arquivo não existir
        geracoes = [rec['generation'] for rec in historico]
        melhores_scores = [rec.get('best_score', 0) for rec in historico]
        medias_scores = [rec.get('avg_score', 0) for rec in historico]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(geracoes, melhores_scores, 'b-', label='Best Clone')
        ax.plot(geracoes, medias_scores, 'r-', label='Population Average')
        ax.set_xlabel('Generation')
        ax.set_ylabel('Score (F1)')
        ax.set_title('Evolution of Performance of Clones')
        ax.legend()
        ax.grid(True)
        
        st.pyplot(fig)
    
    # Visualização da árvore genealógica
    st.subheader("Clone Genealogy Tree")
    
    if os.path.exists("output/visualizations/genealogy_tree.png"):
        st.image("output/visualizations/genealogy_tree.png")
    else:
        st.info("Clone genealogy visualization is not available. Execute the visualization command to generate it.")
        st.code("python -m mestre_dos_clones.cli visualize", language="bash")
    
    # Estatísticas por geração
    st.subheader("Generation Statistics")
    
    geracao_selecionada = st.selectbox(
        "Select a generation to view details:",
        range(1, total_geracoes + 1)
    )
    
    # Encontra os dados da geração selecionada
    dados_geracao = historico[geracao_selecionada - 1]
    
    # Mostra informações da geração
    st.write(f"**Generation {geracao_selecionada}**")
    st.write(f"Population size: {dados_geracao.get('population_size', 'N/A')}")
    st.write(f"Best score: {dados_geracao.get('best_score', 0):.4f}")
    st.write(f"Average score: {dados_geracao.get('avg_score', 0):.4f}")
    
    # Lista de clones da geração
    st.write("**Clones of this generation:**")
    
    # Cria uma tabela de clones
    dados_clones = []
    for clone in dados_geracao.get('clones', []):
        dados_clones.append({
            'Name': clone.get('name', 'N/A'),
            'Algorithm': clone.get('algorithm', 'N/A'),
            'Vectorizer': clone.get('vectorizer', 'N/A'),
            'Score': clone.get('score', 0),
            'Internal Generation': clone.get('generation', 1)
        })
    
    if dados_clones:
        df_clones = pd.DataFrame(dados_clones)
        df_clones = df_clones.sort_values(by='Score', ascending=False)
        st.dataframe(df_clones)
    else:
        st.info("No clone data available for this generation.")

# === Test Clones Page ===
elif pagina == "Test Clones":
    st.header("Test Clones")
    
    # Verifica se o sistema já foi executado
    if not sistema_executado():
        st.warning("⚠️ The system has not been run yet. Go to 'Run Evolution' to get started.")
        st.stop()
    
    st.markdown("""
    In this section, you can test the clones generated by the system.
    Select a generation and a specific clone, and try classifying texts.
    """)
    
    # Layout de colunas para seleção de clone e teste
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Select a Clone")
        
        # Lista de gerações disponíveis
        geracoes = listar_geracoes()
        if not geracoes:
            st.error("No generations found. Execute the system first.")
            st.stop()
        
        geracao_selecionada = st.selectbox(
            "Generation:",
            geracoes
        )
        
        # Lista de clones da geração selecionada
        clones_disponiveis = listar_clones(geracao_selecionada)
        if not clones_disponiveis:
            st.error(f"No clones found in generation {geracao_selecionada}.")
            st.stop()
        
        clone_selecionado = st.selectbox(
            "Clone:",
            clones_disponiveis
        )
        
        # Carrega o clone selecionado
        clone = carregar_clone(geracao_selecionada, clone_selecionado)
        if clone is None:
            st.error("Error loading selected clone.")
            st.stop()
        
        # Mostra informações do clone
        st.subheader("Clone Information")
        st.write(f"**Name:** {clone.name}")
        st.write(f"**Algorithm:** {clone.algorithm_name}")
        st.write(f"**Vectorizer:** {clone.vectorizer_name}")
        st.write(f"**Score:** {clone.performance_score or 0:.4f}")
        st.write(f"**Internal Generation:** {clone.generation}")
        
        if clone.parent_ids:
            st.write(f"**Parents:** {', '.join(clone.parent_ids)}")
        else:
            st.write("**Parents:** None (first generation clone)")
    
    with col2:
        st.subheader("Text Classification")
        
        # Área para digitar texto para classificação
        texto = st.text_area(
            "Enter a text to classify:",
            height=150,
            placeholder="Ex: The product is incredible, exceeded my expectations!"
        )
        
        # Exemplos pré-definidos
        st.markdown("Or choose an example:")
        exemplos = [
            "The service was excellent, I recommend to everyone!",
            "Product of poor quality, broke on first use.",
            "I'm very satisfied with the purchase, it was worth it.",
            "I was disappointed with the service, I don't recommend."
        ]
        
        exemplo_selecionado = st.selectbox(
            "Examples:",
            ["Select an example..."] + exemplos
        )
        
        if exemplo_selecionado != "Select an example...":
            texto = exemplo_selecionado
            st.text_area(
                "Selected Text:",
                value=texto,
                height=100,
                disabled=True
            )
        
        # Botão para classificar
        if st.button("Classify Text"):
            if not texto:
                st.warning("Enter or select a text to classify.")
            else:
                with st.spinner("Classifying..."):
                    sentimento, predicao = classificar_texto(clone, texto)
                    
                    if sentimento:
                        # Mostra resultado com cor
                        if sentimento == "positive":
                            st.success(f"Sentiment: {sentimento.upper()} (Score: {predicao:.4f})")
                        else:
                            st.error(f"Sentiment: {sentimento.upper()} (Score: {1-predicao:.4f})")
                        
                        # Explicação do resultado
                        st.info(f"""
                        The clone "{clone.name}" classified the text as **{sentimento}**.
                        
                        This clone uses the algorithm **{clone.algorithm_name}** with vectorization **{clone.vectorizer_name}**
                        and achieved a score of **{clone.performance_score or 0:.4f}** in tests.
                        """)
        
        # Seção para classificação em lote
        st.subheader("Batch Classification")
        st.markdown("""
        You can also classify multiple texts at once.
        Upload a file with a text per line or enter manually.
        """)
        
        # Opções para entrada em lote
        opcao_lote = st.radio(
            "How do you want to provide the texts?",
            ["Enter manually", "Upload file"]
        )
        
        if opcao_lote == "Enter manually":
            textos_lote = st.text_area(
                "Enter each text in a new line:",
                height=150,
                placeholder="Text 1\nText 2\nText 3"
            )
            
            if st.button("Classify in Batch"):
                if not textos_lote:
                    st.warning("Enter at least one text to classify.")
                else:
                    linhas = textos_lote.strip().split("\n")
                    resultados = []
                    
                    with st.spinner(f"Classifying {len(linhas)} texts..."):
                        for linha in linhas:
                            if linha.strip():
                                sentimento, _ = classificar_texto(clone, linha.strip())
                                resultados.append({
                                    "Text": linha.strip(),
                                    "Sentiment": sentimento
                                })
                    
                    if resultados:
                        df_resultados = pd.DataFrame(resultados)
                        st.dataframe(df_resultados)
                        
                        # Contagem por sentimento
                        positivos = sum(1 for r in resultados if r["Sentiment"] == "positive")
                        negativos = len(resultados) - positivos
                        
                        st.write(f"**Total:** {len(resultados)} texts")
                        st.write(f"**Positives:** {positivos} ({positivos/len(resultados)*100:.1f}%)")
                        st.write(f"**Negatives:** {negativos} ({negativos/len(resultados)*100:.1f}%)")
        else:
            arquivo_lote = st.file_uploader(
                "Upload file with texts:",
                type=["txt", "csv"],
                help="One text per line"
            )
            
            if arquivo_lote is not None:
                try:
                    textos_lote = arquivo_lote.read().decode().strip().split("\n")
                    st.write(f"File loaded: {len(textos_lote)} texts")
                    
                    if st.button("Classify File"):
                        resultados = []
                        
                        with st.spinner(f"Classifying {len(textos_lote)} texts..."):
                            for linha in textos_lote:
                                if linha.strip():
                                    sentimento, _ = classificar_texto(clone, linha.strip())
                                    resultados.append({
                                        "Text": linha.strip(),
                                        "Sentiment": sentimento
                                    })
                        
                        if resultados:
                            df_resultados = pd.DataFrame(resultados)
                            st.dataframe(df_resultados)
                            
                            # Contagem por sentimento
                            positivos = sum(1 for r in resultados if r["Sentiment"] == "positive")
                            negativos = len(resultados) - positivos
                            
                            st.write(f"**Total:** {len(resultados)} texts")
                            st.write(f"**Positives:** {positivos} ({positivos/len(resultados)*100:.1f}%)")
                            st.write(f"**Negatives:** {negativos} ({negativos/len(resultados)*100:.1f}%)")
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")

# === Features Page ===
elif pagina == "Features":
    render_features_page()

# === About Page ===
elif pagina == "About":
    st.header("About the Clone Master Project")
    
    st.markdown("""
    ## Clone Master: AGI that Evolves AGIs
    
    ### Overview
    
    The **Clone Master** is an experimental system that implements an evolutionary approach for the development
    of specialized artificial intelligences. Inspired by the principles of natural evolution and genetic engineering,
    the system uses a multi-agent architecture to create, train, evaluate, and evolve "clones" of AGI without direct
    human intervention.
    
    ### Architecture
    
    The system consists of four main agents that work together:
    
    1. **Intelligence Architect** 🏗️
       - Designs new clones with different architectures
       - Defines parameters, hyperparameters, and model structures
       - Implements variations and mutations in existing clones
    
    2. **Skills Trainer** 🎓
       - Trains each clone with specific data
       - Adjusts models to maximize their performance
       - Applies optimization and regularization techniques
    
    3. **Performance Evaluator** 📊
       - Tests each clone in controlled test scenarios
       - Calculates performance metrics (precision, recall, F1)
       - Identifies strengths and weaknesses of models
    
    4. **Evolutionary Selector** 🧬
       - Applies natural selection principles to clones
       - Determines which models survive for the next generation
       - Implements crossover and mutation for evolution
    
    ### System Flow
    
    1. The cycle starts with the initial generation of clones by the Architect
    2. Each clone is trained by the Trainer with a specific dataset
    3. The Evaluator tests the clones and assigns performance scores
    4. The Selector chooses the best clones and generates new population
    5. The cycle repeats with each generation improving gradually
    
    ### Applications
    
    - **Research**: Exploration of new algorithms and AI architectures
    - **Education**: Demonstration of concepts of evolution and machine learning
    - **Production**: Development of specialized models for specific tasks
    
    ### Team
    
    This project is developed and maintained by enthusiastic researchers and engineers
    in artificial intelligence and evolutionary systems.
    
    ### Contact and Contributions
    
    To learn more about the project, contribute, or report issues, 
    visit the official repository or contact the development team.
    """)
    
    # Technical information
    st.subheader("Technical Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Languages and Frameworks:**
        - Python 3.9+
        - Scikit-learn
        - NumPy/Pandas
        - Matplotlib/Plotly
        - Streamlit
        """)
    
    with col2:
        st.markdown("""
        **System Requirements:**
        - CPU: 2+ cores
        - RAM: 4GB+ recommended
        - Space: 500MB for installation
        - OS: Windows/Mac/Linux
        """)
    
    # Version and license
    st.markdown("---")
    st.caption("Clone Master v1.0.0 | License MIT | © 2023")

# === Settings Page ===
elif pagina == "Settings":
    st.header("⚙️ Settings")
    
    st.markdown("""
    Personalize your experience and adjust the Clone Master system settings
    according to your needs.
    """)
    
    # Tabs to organize settings
    tab1, tab2, tab3 = st.tabs(["System", "Visualization", "Advanced"])
    
    with tab1:
        st.subheader("System Settings")
        
        # Directories and paths
        st.write("**Directories and Paths**")
        
        data_dir = st.text_input(
            "Data Directory:",
            value="data/",
            help="Location where data files are stored"
        )
        
        output_dir = st.text_input(
            "Output Directory:",
            value="output/",
            help="Location where results will be saved"
        )
        
        # Performance settings
        st.write("**Performance Settings**")
        
        enable_multiprocessing = st.toggle(
            "Enable Multiprocessing",
            value=True,
            help="Use multiple cores of the processor to accelerate evolution"
        )
        
        if enable_multiprocessing:
            num_workers = st.slider(
                "Number of Workers:",
                min_value=2,
                max_value=16,
                value=4,
                help="Number of parallel processes (recommended: number of CPU cores)"
            )
        
        # Cache settings
        use_cache = st.toggle(
            "Enable Result Cache",
            value=True,
            help="Store intermediate results to avoid recalculations"
        )
        
        if use_cache:
            cache_size = st.select_slider(
                "Cache Size:",
                options=["Small", "Medium", "Large", "Unlimited"],
                value="Medium",
                help="Determines how much disk space will be used for cache"
            )
    
    with tab2:
        st.subheader("Visualization Settings")
        
        # Plot style
        st.write("**Plot Style**")
        
        plot_theme = st.selectbox(
            "Plot Theme:",
            ["default", "darkgrid", "whitegrid", "dark", "white", "ticks"],
            index=0,
            help="Visual style for generated plots"
        )
        
        plot_palette = st.selectbox(
            "Plot Color Palette:",
            ["viridis", "plasma", "inferno", "magma", "cividis", "muted", "pastel"],
            index=0,
            help="Color scheme for plots"
        )
        
        # Display information
        st.write("**Display Information**")
        
        show_metrics = st.multiselect(
            "Metrics to Display:",
            ["Accuracy", "Precision", "Recall", "F1", "AUC", "Log Loss"],
            default=["Accuracy", "F1"],
            help="Which metrics should be displayed in results"
        )
        
        decimal_places = st.slider(
            "Decimal Places:",
            min_value=2,
            max_value=6,
            value=4,
            help="Number of decimal places to display numeric values"
        )
    
    with tab3:
        st.subheader("Advanced Settings")
        
        # Algorithms
        st.write("**Allowed Algorithms**")
        
        allowed_algorithms = st.multiselect(
            "Allowed Algorithms:",
            ["Logistic Regression", "Naive Bayes", "SVM", "Decision Tree", 
             "Random Forest", "Gradient Boosting", "KNN", "Neural Network"],
            default=["Logistic Regression", "Naive Bayes", "SVM", "Random Forest"],
            help="Algorithms that the Architect can use to create clones"
        )
        
        # Evolution
        st.write("**Evolution Parameters**")
        
        mutation_rate = st.slider(
            "Mutation Rate:",
            min_value=0.01,
            max_value=0.5,
            value=0.1,
            step=0.01,
            help="Probability of mutation in a clone (0.1 = 10%)"
        )
        
        crossover_rate = st.slider(
            "Crossover Rate:",
            min_value=0.1,
            max_value=0.9,
            value=0.7,
            step=0.05,
            help="Probability of crossover between clones (0.7 = 70%)"
        )
        
        # System reset
        st.write("**System Reset**")
        
        st.warning("⚠️ These options affect saved system data.")
        
        if st.button("Clear Cache", type="secondary"):
            st.info("Cache would be cleared here (simulation)")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Reset Settings", type="secondary"):
                st.info("Settings would be reset (simulation)")
        
        with col2:
            if st.button("Reset System", type="secondary"):
                st.error("This action would delete all clones, results, and history")
                # In real implementation, this would be a confirmation modal

# Footer of the application (present in all pages)
st.markdown("---")
st.caption("Developed using Streamlit")
