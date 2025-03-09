import openai
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import shap
import pandas as pd
import numpy as np

# Configuração da API da OpenAI
openai.api_key = "sk-YH3BHhH_m25dBKXn6Z8bUeeeFnxZhNHWjeQdP4WM63T3BlbkFJ4jariw81lCYCbUuYEHWtnvMEttrcnb8JsAtRwB_jIA"

# Criar um modelo OpenAI atualizado
chat = ChatOpenAI(model="gpt-4", temperature=0.7)

# Função para gerar explicação do agente
def gerar_explicacao_agent(aluno_escolhido, alunos, shap_values, variaveis):
    
      # Verifica se o aluno existe no DataFrame
    if aluno_escolhido not in alunos.index:
        return "Aluno não encontrado na base de dados."
    
    
   # Obtém o índice do aluno no DataFrame
    idx_aluno = alunos.index.get_loc(aluno_escolhido)

    # Obtém os valores SHAP do aluno
    shap_aluno = shap_values[idx_aluno]

    # Identificar as 5 variáveis mais influentes
    top_features_idx = np.argsort(-np.abs(shap_aluno.values))[:5]
    top_features = variaveis.columns[top_features_idx]
    top_shap_values = shap_aluno.values[top_features_idx]

    # Explicações pré-definidas para variáveis comuns
    explicacoes = {
        "Índice de Eficiência Acadêmica": "Este índice mede a eficiência do aluno no cumprimento de requisitos acadêmicos.",
        "Coeficiente de Rendimento": "Reflete o desempenho acadêmico do aluno com base nas notas.",
        "Índice de Eficiência em Carga Horária": "Mede o quanto o aluno aproveita suas horas de estudo de maneira eficaz.",
        "Média de Conclusão": "Representa a média de tempo que o aluno leva para concluir suas disciplinas.",
        "Tempo de Permanência": "Indica o tempo total que o aluno passou no curso."
    }

    # Sugestões de melhoria para cada fator
    sugestoes = {
        "Índice de Eficiência Acadêmica": "Os alunos podem melhorar esse índice com técnicas de estudo eficazes.",
        "Coeficiente de Rendimento": "Manter registros detalhados de desempenho pode ajudar a melhorar este coeficiente.",
        "Índice de Eficiência em Carga Horária": "Recomenda-se ajudar os alunos a gerenciar melhor seu tempo de estudo.",
        "Média de Conclusão": "Incentivar os alunos a concluir tarefas dentro do prazo pode otimizar sua trajetória acadêmica.",
        "Tempo de Permanência": "Oferecer suporte acadêmico pode ajudar alunos a manter um progresso consistente."
    }

    shap_data = {}  # Inicializa o dicionário vazio

    for i in range(5):  # Loop para preencher os 5 principais fatores
        shap_data[f"feature_{i+1}"] = top_features[i]
        shap_data[f"shap_{i+1}"] = round(top_shap_values[i], 3)
        shap_data[f"explicacao_{i+1}"] = explicacoes.get(top_features[i], "Sem explicação disponível")
        shap_data[f"sugestao_{i+1}"] = sugestoes.get(top_features[i], "Sem sugestão disponível")

    prompt_template = """
    📊 **Análise de Desempenho Acadêmico para o aluno**

        Os valores SHAP ajudam a entender o impacto das variáveis no desempenho do aluno e na previsão do modelo. 
        Nesta análise, identificamos os **cinco principais fatores** que influenciam o resultado:

        🔎 **Principais fatores que impactam a previsão**:

        1️⃣ **{feature_1} ({shap_1})**: {explicacao_1}
        2️⃣ **{feature_2} ({shap_2})**: {explicacao_2}
        3️⃣ **{feature_3} ({shap_3})**: {explicacao_3}
        4️⃣ **{feature_4} ({shap_4})**: {explicacao_4}
        5️⃣ **{feature_5} ({shap_5})**: {explicacao_5}

        📌 **Interpretação dos valores**:  
        Valores **positivos** indicam fatores que contribuem para um **bom desempenho** do aluno, enquanto valores **negativos** sugerem possíveis **riscos de evasão ou baixo rendimento**.

        ---

        📌 **Sugestões para melhorar o desempenho**:

        🔹 **{feature_1}**: {sugestao_1}  
        🔹 **{feature_2}**: {sugestao_2}  
        🔹 **{feature_3}**: {sugestao_3}  
        🔹 **{feature_4}**: {sugestao_4}  
        🔹 **{feature_5}**: {sugestao_5}  

        Seja claro e direto ao formular a resposta, garantindo que professores e gestores possam entender rapidamente as implicações e tomar decisões eficazes.
        """

    response = chat.invoke([
        SystemMessage(content="Você é um assistente de IA especialista em análise educacional."),
        HumanMessage(content=prompt_template.format(**shap_data))
    ])

    return response.content
