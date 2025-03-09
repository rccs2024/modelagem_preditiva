import joblib
import pandas as pd
from colorama import Fore, Style




def Preprocessamento(df):
    modelo = joblib.load("Model/model_xgboost_prever_conclusao_13_02.pkl")
    scaler = joblib.load("Model/scaler_numerico_13_02_B.pkl")
    label_encoders = joblib.load('Model/label_encoders_13_02.pkl')

    var_excluir = ['Data_Nascimento','Cidade_Estado','CEP','Ano_Ingresso','Periodo_Ingresso','Forma_Ingresso','Ano_Periodo_Saida','Tipo_Saida',
               'Prazo_Integralizacao','Prazo_Integralizacao_Ano', 'Prazo_Integralizacao_Semestre','Ano_Saida','Semestre_Saida',
               'Tempo_permanencia_Meses','Data_Ingresso','Idade_Ingresso','Curso']

    df_modelagem = df.drop(columns=var_excluir)

   # 1.2. Codificação das variáveis categóricas
    variaveis_categoricas = ['Tipo_Raca', 'Tipo_Rede_Ensino', 'Faixa_Etaria','Sexo','Região_Origem']
    for col in variaveis_categoricas:
        if col in label_encoders:
            df_modelagem[col] = label_encoders[col].transform(df_modelagem[col])

    # Normalização das variáveis numéricas
    numeric_features = df_modelagem.select_dtypes(include=['float64']).columns
    df_modelagem[numeric_features] = scaler.transform(df_modelagem[numeric_features])

    # Ajustar colunas do modelo treinado
    X = df_modelagem.drop(columns=['Status'])
   

    return X

# Função para prever evasão/conclusão
def prever_evasao(df_aluno):
    modelo = joblib.load("Model/model_xgboost_prever_conclusao_13_02.pkl")
    df_preparado = Preprocessamento(df_aluno)
    probabilidades = modelo.predict_proba(df_preparado)

    threshold_otimizado = 0.44
    resultados = []

    for i, probs in enumerate(probabilidades):
        prob_evasao = probs[0]
        prob_conclusao = probs[1]
        status_predito = 'CONCLUÍDO' if prob_conclusao >= threshold_otimizado else 'DESLIGADO'
        situacao_real = str(df_aluno.iloc[i]["Status"]).strip().upper()
        resultado = "✅ ACERTOU" if situacao_real == status_predito else "❌ ERROU"

        resultados.append({
            "Nome": df_aluno.iloc[i].name,
            "Situação Atual": situacao_real,
            "Probabilidade de Evasão": f"{prob_evasao:.2%}",
            "Probabilidade de Conclusão": f"{prob_conclusao:.2%}",
            "Classificação do Modelo": status_predito,
            "Resultado": resultado
        })

    return resultados