import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from functions import Preprocessamento, prever_evasao
from agent import gerar_explicacao_agent
import time

# Criando Interface com Streamlit
st.set_page_config(page_title="Previsão de Conclusão de Alunos", layout="wide")
st.title("🎓 Previsão de Conclusão de Alunos")

# Upload do arquivo CSV
st.sidebar.header("📂 Carregar Base de Dados")
uploaded_file = st.sidebar.file_uploader("Faça o upload do arquivo CSV", type=["csv"])

if uploaded_file is not None:
    # Carregar os dados
    df = pd.read_csv(uploaded_file, delimiter=",")
    df.set_index("Nome", inplace=True)
    df2 = df.copy()

    # Filtrar apenas alunos com status conhecido
    alunos = df[df["Status"].isin(["DESLIGADO", "CONCLUÍDO"])]

    st.write("### 📋 Dados Carregados:")
    st.dataframe(alunos)

    # Selecionar alunos para prever
    st.sidebar.subheader("🔎 Selecione a quantidade de alunos para prever")
    qtd_alunos = st.sidebar.slider("Número de alunos:", min_value=1, max_value=len(alunos), value=10)

    if st.sidebar.button("📊 Executar Previsão"):
        # Salvar os resultados na sessão para evitar recomputação
        st.session_state["alunos_selecionados"] = alunos[:qtd_alunos]
        st.session_state["resultados"] = prever_evasao(st.session_state["alunos_selecionados"])

        modelo = joblib.load("Model/model_xgboost_prever_conclusao_13_02.pkl")
        variaveis = Preprocessamento(df2)
        
        explainer = shap.Explainer(modelo, variaveis)
        shap_values = explainer(variaveis)

        st.session_state["shap_values"] = shap_values
        st.session_state["variaveis"] = variaveis

    # Exibir os resultados das previsões
    if "resultados" in st.session_state:
        tab1, tab2 = st.tabs(["Previsões de Alunos", "Previsão Individual"])
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.write("### 📈 Resultados das Previsões:")
                df_resultado = pd.DataFrame(st.session_state["resultados"])
                st.dataframe(df_resultado)
                
            with col2:
                modelo = joblib.load("Model/model_xgboost_prever_conclusao_13_02.pkl")
                variaveis = Preprocessamento(df2)
                # Criar explainer e calcular valores SHAP
                explainer = shap.Explainer(modelo, variaveis)
                shap_values = explainer(variaveis)
                
                # Plot SHAP Waterfall
                fig, ax = plt.subplots()
                shap.summary_plot(shap_values, variaveis)
                st.pyplot(fig)
                
            
            
            with col1:
            # Comparação de Importância Global
                st.subheader("📊 Comparação entre 'Concluídos' e 'Desligados'")
                df_shap = pd.DataFrame(st.session_state["shap_values"].values, columns=st.session_state["variaveis"].columns)
                df_shap["Status"] = df["Status"].values  
                df_shap = df_shap[df_shap["Status"].isin(["CONCLUÍDO", "DESLIGADO"])]
                shap_mean = df_shap.groupby("Status").mean().T

                fig, ax = plt.subplots(figsize=(10, 6))
                shap_mean.plot(kind="barh", ax=ax)
                ax.set_title("Comparação de Importância das Variáveis entre Concluídos e Desligados")
                ax.set_xlabel("Média dos valores SHAP")
                st.pyplot(fig)
            

        with tab2:
            st.write("### 🔍 Análise Individual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                aluno_escolhido = st.selectbox("Selecione um aluno", df_resultado["Nome"])
                    # Inicializar controle de estado
                if "ultimo_aluno_explicado" not in st.session_state:
                    st.session_state["ultimo_aluno_explicado"] = None  

                if "explicacao_agent" not in st.session_state:
                    st.session_state["explicacao_agent"] = None  

                # Se o aluno mudou, resetar explicação anterior
                if aluno_escolhido != st.session_state["ultimo_aluno_explicado"]:
                    st.session_state["explicacao_agent"] = None  
                    st.session_state["ultimo_aluno_explicado"] = aluno_escolhido  
            
            with col2:                            
                              
                curso_aluno = df.loc[aluno_escolhido, "Curso"]
                df_curso = df[df["Curso"] == curso_aluno]
                df_numerico = df_curso.select_dtypes(include=["number"])
                df_numerico = df_numerico.set_index(df_curso.index)
                mean_values = df_numerico.groupby(df_curso["Status"]).mean()               

                
                colunas_relevantes = [
                            "Coeficiente_de_Rendimento", "Média_de_Conclusão",
                            "Índice_de_Eficiência_Acadêmica", "Índice_de_Eficiência_em_Carga_Horária",
                            "Índice_de_Eficiência_em_Períodos_Letivos","Tempo_permanencia_Meses","Idade_Ingresso"
                        ]
                

                # Garantir que o aluno está no DataFrame e selecionar as colunas certas
                if aluno_escolhido:
                    df_resultado = pd.DataFrame(st.session_state["resultados"])  
                    # Garantir que o aluno existe no dataframe antes de buscar a previsão
                    if aluno_escolhido in df_resultado["Nome"].values:
                        previsao = df_resultado[df_resultado["Nome"].str.strip() == aluno_escolhido.strip()]["Classificação do Modelo"].values[0]
                    else:
                        previsao = "NÃO ENCONTRADO"
                    
                   # Determinar o indicador visual
                    if previsao == "CONCLUÍDO":
                        indicador = "✅ Concluído"
                        cor = "green"
                    else:
                        indicador = "❌ Desligado"
                        cor = "red"
                    
                    situacao_real = df.loc[aluno_escolhido, "Status"]  # Pegando a situação real do aluno
                    # Verificar se a previsão estava correta
                    if previsao == situacao_real:
                        resultado_previsao = "✅ Acertou"
                        cor_resultado = "green"
                    else:
                        resultado_previsao = "❌ Errou"
                        cor_resultado = "red"

                                        
                    # Exibir as informações acima da tabela de comparação
                    st.markdown(f"### 🎓 Curso: **{curso_aluno}**")
                    st.markdown(f"<h3 style='color:{cor};'>📌 Previsão do Modelo: {indicador}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<h4 style='color:{cor_resultado};'>🎯 Resultado: {resultado_previsao}</h4>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
               
                                                            
                    st.markdown("📊 Comparação com o Grupo")
                    dados_aluno = df_curso.loc[aluno_escolhido, colunas_relevantes] # Selecionar linha do aluno

                    df_comparacao = pd.DataFrame({
                        "Aluno": dados_aluno,
                        "Média Concluídos": mean_values.loc["CONCLUÍDO", colunas_relevantes] if "CONCLUÍDO" in mean_values.index else None,
                        "Média Desligados": mean_values.loc["DESLIGADO", colunas_relevantes] if "DESLIGADO" in mean_values.index else None
                    })

                    st.dataframe(df_comparacao)
                else:
                    st.error(f"⚠️ O aluno '{aluno_escolhido}' não foi encontrado no grupo '{curso_aluno}'.")
            
            with col1:
                if aluno_escolhido:
                    dados_aluno = df.loc[aluno_escolhido]
                    idx_aluno = df.index.get_loc(aluno_escolhido)
                    shap_aluno = st.session_state["shap_values"][idx_aluno]

                    st.subheader("📊 Explicação SHAP Waterfall Plot")
                    shap_exp = shap.Explanation(
                        values=shap_aluno.values, 
                        base_values=shap_aluno.base_values, 
                        data=pd.Series(dados_aluno, index=st.session_state["variaveis"].columns),
                        feature_names=st.session_state["variaveis"].columns
                    )

                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap_exp, max_display=10)
                    st.pyplot(fig)
                    
                    # Criar botão para exibir explicação do agente
             # --- 🔹 Ajuste para Gerar Explicação Apenas Quando Clicado ---
    
         

            # Criar botão para exibir explicação do agente
            if st.button("📄 Explicar Análise"):
                with st.spinner("Gerando explicação..."):
                    explicacao = gerar_explicacao_agent(
                        aluno_escolhido, 
                        st.session_state["alunos_selecionados"], 
                        st.session_state["shap_values"], 
                        st.session_state["variaveis"]
                    )
                    st.session_state["explicacao_agent"] = explicacao  # Armazena para evitar recomputação

            # Mostrar explicação se estiver disponível
            if st.session_state["explicacao_agent"]:
                with st.expander("📄 Explicação da Análise", expanded=True):
                    placeholder = st.empty()  # Criar espaço vazio para atualizar dinamicamente
                    texto = st.session_state["explicacao_agent"]
                    
                    # Exibir o texto simulando digitação
                    texto_exibido = ""
                    for letra in texto:
                        texto_exibido += letra
                        placeholder.markdown(texto_exibido + "▌")  # Simula um cursor piscando
                        time.sleep(0.01)  # Ajuste para controlar a velocidade
                    
                    placeholder.markdown(texto_exibido)  # Remove o cursor no final
