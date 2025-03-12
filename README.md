# 🧠 Explicabilidade de Modelos Preditivos na Evasão Acadêmica: Um Estudo de Caso

## 📌 Objetivo

Este projeto tem como objetivo construir um modelo de machine learning capaz de prever a evasão acadêmica de estudantes de graduação em uma universidade pública brasileira. Utilizando o algoritmo **XGBoost** e a técnica **SHAP** para interpretabilidade, a proposta é oferecer **insights acionáveis para gestores educacionais** com base no desempenho acadêmico e no perfil dos alunos.

---

## 📂 Estrutura do Projeto

| Arquivo | Descrição |
|--------|-----------|
| `1 - Prepara_dados.ipynb` | Tratamento de valores ausentes, ajustes nas variáveis e criação da base final. |
| `2 - Exploracao_Dados.ipynb` | Análise exploratória com comparações entre alunos concluintes e evadidos. |
| `3 - Pipeline_Modelagem_Preditiva_XGBoost.ipynb` | Criação do pipeline de modelagem com SMOTE e avaliação de modelos. |
| `4 - Avaliacao_e_Explicabilidade_Modelo_XGBoost.ipynb` | Aplicação do modelo em nova base e explicabilidade com SHAP. |

---

## 📊 Tecnologias e Bibliotecas

- Python 3.10+
- Pandas, Numpy
- Matplotlib, Seaborn, Plotly
- Scikit-learn
- XGBoost
- SHAP
- imbalanced-learn (SMOTE)
- Joblib

---

## 📑 Sobre os Dados

Os dados utilizados neste projeto foram extraídos do sistema acadêmico da instituição e contêm informações sensíveis, portanto **não podem ser compartilhados publicamente**.

Contudo, a base inclui variáveis como:

- **Dados demográficos**: Sexo, Tipo de escola (pública/privada), Raça, Faixa etária, Região de origem.
- **Desempenho acadêmico**: Coeficiente de Rendimento, Média de Conclusão, Índices de Eficiência Acadêmica.
- **Informações temporais**: Ano de ingresso, Tempo de permanência, Prazo de integralização.
- **Situação final**: Concluído ou Desligado (evasão).

🔗 O dicionário de dados completo está disponível nos notebooks do projeto.

---

## ⚙️ Metodologia

O projeto seguiu a metodologia **CRISP-DM**, com foco nas etapas:

- **Preparação dos dados**: Limpeza, criação de variáveis derivadas e recodificação de status.
- **Análise exploratória**: Investigação dos perfis dos alunos evadidos e concluintes.
- **Modelagem preditiva**: Testes com Random Forest, Regressão Logística e XGBoost, com balanceamento via SMOTE.
- **Explicabilidade com SHAP**: Interpretação das previsões para identificar variáveis com maior impacto.

---

## 📈 Resultados

- **Melhor modelo**: XGBoost
- **Acurácia na base de teste**: **93,19%**
- **Principais variáveis preditoras**:
  - Índice de Eficiência Acadêmica
  - Coeficiente de Rendimento
  - Média de Conclusão
  - Tipo de Escola (Pública/Privada)
- **Explicabilidade com SHAP**:  
  - Gráficos de *summary*, *waterfall* e *dependence* explicam o impacto de cada atributo na decisão.

> O modelo é interpretável, escalável e pode ser usado como base para um sistema de apoio à decisão institucional.
