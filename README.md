# Customer Churn Prediction (Previsão de Cancelamento de Clientes)

Este projeto de **Classificação Binária** foca em prever e entender o cancelamento de clientes (*Churn*) em uma empresa de telecomunicações.

O objetivo é construir um modelo preditivo robusto que auxilie o time de retenção a identificar clientes em alto risco, permitindo intervenções proativas e direcionadas.

---

## Metodologia

1.  **Análise Exploratória (EDA):** Identificação de desbalanceamento de classes ($\approx 26.5\%$ de Churn) e *insights* iniciais (Churn é maior em clientes de curto prazo).
2.  **Pré-processamento:** Tratamento de valores nulos (imputação pela mediana em `TotalCharges`), codificação de colunas binárias e aplicação de **One-Hot Encoding** para variáveis categóricas.
3.  **Modelagem e Avaliação:** Teste de múltiplos modelos (Regressão Logística, Random Forest, XGBoost, LightGBM). Foi priorizada a métrica **Recall** para a classe minoritária (Churn), visando minimizar **Falsos Negativos** (clientes perdidos).
4.  **Otimização:** Uso de **Grid Search** focado no `scoring='recall'` e técnicas de desbalanceamento (`scale_pos_weight` ou `class_weight='balanced'`).

---

## Fonte dos Dados

* **Fonte de Dados:** [Telco Customer Churn Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Resultados Finais e *Insights*

O **XGBoost Classifier Otimizado** foi o modelo vencedor, pois alcançou o melhor desempenho na métrica de negócio crucial.

| Métrica | Desempenho | Comentário |
| :--- | :--- | :--- |
| **Recall (Churn=1)** | **0.82** | O modelo identifica corretamente **82%** dos clientes que realmente cancelam (Minimização de Falsos Negativos). |
| **AUC-ROC** | **0.8431** | Excelente poder de discriminação geral. |

### Principais Fatores de Risco (*Feature Importance*)

A análise de importância das *features* (*Feature Importance*) revelou os principais *drivers* de *Churn*:

1.  **Contrato de Um Ano (`Contract_One year`):** Principal fator preditivo de risco.
2.  **Contrato de Dois Anos (`Contract_Two year`):** Principal fator preditivo de **retenção**.
3.  **Serviço de Fibra Óptica (`InternetService_Fiber optic`):** Forte indicador de risco de Churn, sugerindo problemas de satisfação nesse serviço.
4.  **Clientes Sem Internet (`InternetService_No`):** Forte indicador de baixa propensão ao cancelamento.

---

## Próximos Passos

O modelo otimizado foi serializado (`models/xgb_model_optimized.pkl`). Próximas fases sugeridas:

* **Deploy:** Construção de um serviço de inferência em tempo real (ex: FastAPI) e empacotamento com Docker.
* **Monitoramento:** Implementação de ferramentas (ex: MLflow) para monitorar o desempenho do modelo em produção e detectar *drift*.