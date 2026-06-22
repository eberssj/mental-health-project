# Mental Health Project

Projeto acadêmico de Machine Learning para classificação de condições de saúde mental a partir de duas abordagens complementares:

- dados estruturados de sintomas;
- textos de relatos relacionados à saúde mental.

O projeto compara modelos tradicionais, baseline textual com TF-IDF e uma abordagem semântica com Sentence-BERT local.

> **Aviso importante:** este projeto tem finalidade acadêmica e experimental. Ele não deve ser usado como ferramenta de diagnóstico médico, triagem clínica ou tomada de decisão em saúde.

## Objetivo

O objetivo é avaliar técnicas de classificação supervisionada em dois tipos de dados:

1. **Sintomas estruturados:** classificação multiclasse usando variáveis binárias de sintomas.
2. **Textos livres:** classificação de relatos usando representação lexical e representação semântica.

A avaliação prioriza métricas adequadas para problemas multiclasse e com possíveis diferenças entre classes, principalmente **F1-score macro**.

## Principais decisões metodológicas

- Não é usada reamostragem sintética com SMOTE.
- O pipeline estruturado usa validação cruzada com `scoring="f1_macro"`.
- O pipeline textual compara:
  - **baseline lexical:** TF-IDF + XGBoost;
  - **modelo semântico:** Sentence-BERT + Logistic Regression.
- O tratamento de diferenças entre classes é feito por pesos (`class_weight` ou `sample_weight`), sem criar exemplos artificiais.
- A acurácia não é usada como métrica principal, pois pode esconder baixo desempenho em classes menores.

## Estrutura do projeto

```text
mental-health-project/
  data/
    illness_dataset.csv
    Mental Health Disorder Detection Dataset.csv

  exports/
    embeddings/
    models/
    plots/

  notebooks/
    01_analise_exploratoria.ipynb
    02_analise_exploratoria.ipynb
    03_nlp_text_analysis.ipynb
    04_nlp_text_analysis.ipynb
    05_nlp_text_analysis.ipynb

  src/
    preprocessing.py
    model.py
    evaluation.py

  main.py
  requirements.txt
  README.md
```

## Bases de dados

### `data/illness_dataset.csv`

Base estruturada usada no primeiro pipeline.

- Quantidade verificada no projeto: **8.304 registros**.
- Variáveis preditoras: **185 colunas** de sintomas.
- Variável alvo: `Disease`.
- Número de classes: **22 condições**.

### `data/Mental Health Disorder Detection Dataset.csv`

Base textual usada no pipeline de NLP.

- Quantidade verificada no projeto após remoção de nulos: **11.272 textos**.
- Coluna de texto: `body`.
- Coluna alvo: `category`.
- Número de classes: **7 categorias**.

## Pipelines implementados

### 1. Pipeline estruturado

Arquivo principal: `main.py`  
Funções auxiliares: `src/preprocessing.py`, `src/model.py`, `src/evaluation.py`

Etapas:

1. Carrega `data/illness_dataset.csv`.
2. Separa `Disease` como variável alvo.
3. Codifica as classes com `LabelEncoder`.
4. Divide treino e teste com `train_test_split`, usando estratificação.
5. Gera matriz de correlação das principais variáveis.
6. Treina e ajusta três modelos com `GridSearchCV`:
   - Logistic Regression;
   - Random Forest;
   - XGBoost.
7. Avalia os modelos com precision, recall, F1-score, F1-macro, matriz de confusão e ROC-AUC quando aplicável.
8. Salva o melhor pipeline estruturado.

O pipeline de cada modelo usa:

```text
PCA(n_components=0.95) -> Classificador
```

### 2. Pipeline textual com TF-IDF

Este é o baseline lexical de NLP.

Etapas:

1. Carrega `data/Mental Health Disorder Detection Dataset.csv`.
2. Remove registros sem texto ou categoria.
3. Codifica as classes com `LabelEncoder`.
4. Vetoriza os textos com `TfidfVectorizer`.
5. Treina um modelo XGBoost com pesos balanceados.
6. Exporta relatório de métricas e matriz de confusão.

Configuração principal do TF-IDF:

```text
max_features=5000
ngram_range=(1, 2)
min_df=5
```

### 3. Pipeline textual com Sentence-BERT

Este é o pipeline semântico principal.

Modelo usado:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Etapas:

1. Converte os textos em embeddings semânticos.
2. Normaliza os embeddings.
3. Usa cache local dos vetores para evitar recomputação.
4. Treina `LogisticRegression` com `class_weight="balanced"`.
5. Exporta relatório de métricas e matriz de confusão.

O modelo Sentence-BERT gera vetores densos de **384 dimensões** por texto.

## Instalação

No Windows com PowerShell, dentro da pasta do projeto:

```powershell
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o ambiente virtual já existir:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Como executar

Executar todos os pipelines:

```powershell
.\venv\Scripts\python.exe main.py
```

Executar apenas o pipeline estruturado:

```powershell
.\venv\Scripts\python.exe main.py structured
```

Executar apenas o pipeline de NLP:

```powershell
.\venv\Scripts\python.exe main.py nlp
```

## Saídas geradas

Os resultados são salvos na pasta `exports/`.

### Modelos

```text
exports/models/
  best_structured_pipeline.pkl
  structured_encoder.pkl
  nlp_model.pkl
  nlp_tfidf_xgb_model.pkl
  nlp_vectorizer.pkl
  nlp_tfidf_vectorizer.pkl
  nlp_semantic_classifier.pkl
  nlp_semantic_config.pkl
  nlp_encoder.pkl
```

### Gráficos

```text
exports/plots/
  structured_correlation.png
  cm_structured_LogisticRegression.png
  cm_structured_RandomForest.png
  cm_structured_XGBoost.png
  cm_nlp_tfidf_xgb.png
  cm_nlp_sentence_bert.png
```

### Cache de embeddings

```text
exports/embeddings/
```

Os arquivos `.npy` dessa pasta armazenam embeddings já calculados. Eles podem ser apagados com segurança, pois serão recriados na próxima execução do pipeline de NLP.

## Métricas utilizadas

O projeto reporta:

- precision;
- recall;
- F1-score por classe;
- F1-score macro;
- matriz de confusão;
- ROC-AUC como métrica secundária quando aplicável.

O **F1-score macro** é especialmente importante porque calcula a média das classes dando o mesmo peso para cada uma. Isso reduz o risco de uma classe majoritária dominar a avaliação.

## Notebooks

Os notebooks registram etapas exploratórias e experimentais:

- `01_analise_exploratoria.ipynb`: análise exploratória inicial.
- `02_analise_exploratoria.ipynb`: continuidade da análise estruturada.
- `03_nlp_text_analysis.ipynb`: exploração textual inicial.
- `04_nlp_text_analysis.ipynb`: baseline TF-IDF + XGBoost.
- `05_nlp_text_analysis.ipynb`: experimento principal com Sentence-BERT.

Para justificar a evolução da representação textual, o notebook mais relevante é o `05_nlp_text_analysis.ipynb`, pois compara a ideia de representação baseada em termos com uma representação semântica por embeddings.

## Verificações rápidas

Verificar se os arquivos Python compilam:

```powershell
.\venv\Scripts\python.exe -m py_compile main.py src\preprocessing.py src\model.py src\evaluation.py
```

Verificar dependências instaladas:

```powershell
.\venv\Scripts\python.exe -m pip check
```

Testar o Sentence-BERT local:

```powershell
.\venv\Scripts\python.exe -c "from src.preprocessing import get_semantic_embeddings; print(get_semantic_embeddings(['I feel anxious every day'], show_progress_bar=False).shape)"
```

Resultado esperado:

```text
(1, 384)
```

Na primeira execução, o modelo público será baixado do Hugging Face. Depois, ele fica armazenado no cache local.

## Arquivos de configuração

### `.gitignore`

O projeto ignora arquivos locais e temporários, como:

- ambiente virtual `venv/`;
- cache Python `__pycache__/`;
- arquivos `.pyc`;
- variáveis locais `.env`;
- arquivos auxiliares de merge.

### `requirements.txt`

Lista as bibliotecas necessárias para executar os scripts, notebooks, modelos clássicos, XGBoost e Sentence-BERT.

## Observações finais

- Não é necessário configurar token do Hugging Face para baixar o modelo público usado neste projeto.
- O pipeline de NLP pode demorar mais na primeira execução por causa do download do modelo e da geração dos embeddings.
- Os resultados salvos em `exports/` documentam os artefatos gerados na execução final.
- O projeto preserva uma separação simples entre pré-processamento, treinamento e avaliação, facilitando a leitura e a apresentação acadêmica.
