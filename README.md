# Mental Health Project

Projeto de classificacao de transtornos/condicoes de saude mental usando duas fontes de dados:

- dados estruturados de sintomas;
- textos de relatos relacionados a saude mental.

O projeto foi ajustado para evitar reamostragem sintetica desnecessaria, usar metricas mais adequadas que acuracia e incluir uma representacao semantica local baseada em Hugging Face/Sentence-BERT.

> Este projeto tem finalidade academica e experimental. Ele nao deve ser usado como ferramenta de diagnostico medico.

## Principais mudancas metodologicas

- SMOTE foi removido dos experimentos estruturados e textuais.
- A avaliacao prioriza precision, recall e F1-score, especialmente F1-macro.
- O pipeline estruturado usa validacao cruzada com `scoring="f1_macro"`.
- O pipeline textual agora compara:
  - baseline lexical: TF-IDF + XGBoost;
  - proposta semantica: Sentence-BERT local + Logistic Regression.
- O tratamento de diferencas entre classes e feito por pesos (`class_weight` ou `sample_weight`), sem criar exemplos sinteticos.

## Estrutura

```text
data/
  illness_dataset.csv
  Mental Health Disorder Detection Dataset.csv

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
```

## Pipelines

### 1. Dados estruturados

O pipeline estruturado carrega `data/illness_dataset.csv`, separa treino e teste de forma estratificada e avalia multiplos classificadores:

- Logistic Regression;
- Random Forest;
- XGBoost.

A busca de hiperparametros usa `GridSearchCV` com F1-macro, sem SMOTE.

### 2. Textos com baseline TF-IDF

O baseline textual usa `TfidfVectorizer` com n-gramas e XGBoost. Ele foi mantido como comparacao porque representa uma abordagem lexical: considera frequencia de palavras/termos, mas nao captura semantica profunda.

### 3. Textos com embeddings semanticos

A versao principal de NLP usa o modelo local:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Esse modelo gera embeddings densos de 384 dimensoes para cada texto. Depois, os vetores sao classificados com Logistic Regression balanceada.

Os embeddings sao salvos em cache em:

```text
exports/embeddings/
```

Assim, a primeira execucao pode demorar mais porque baixa o modelo e gera os vetores. Execucoes posteriores reutilizam o cache quando o split e o modelo forem os mesmos.

## Instalacao

No Windows/PowerShell:

```powershell
cd D:\fatec\mental-health-project
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

Se for criar um ambiente virtual novo:

```powershell
cd D:\fatec\mental-health-project
python -m venv venv
.\venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Como executar

Rodar tudo:

```powershell
.\venv\Scripts\python.exe main.py
```

Rodar apenas o pipeline estruturado:

```powershell
.\venv\Scripts\python.exe main.py structured
```

Rodar apenas o pipeline de NLP:

```powershell
.\venv\Scripts\python.exe main.py nlp
```

## Teste rapido do Hugging Face local

Use este comando para confirmar que o Sentence-BERT esta funcionando:

```powershell
.\venv\Scripts\python.exe -c "from src.preprocessing import get_semantic_embeddings; print(get_semantic_embeddings(['I feel anxious every day'], show_progress_bar=False).shape)"
```

O resultado esperado e semelhante a:

```text
(1, 384)
```

Na primeira execucao, o modelo sera baixado do Hugging Face. Depois ele fica no cache local.

## Saidas geradas

Modelos e configuracoes:

```text
exports/models/
```

Graficos e matrizes de confusao:

```text
exports/plots/
```

Embeddings semanticos em cache:

```text
exports/embeddings/
```

Arquivos principais da versao semantica:

- `exports/models/nlp_semantic_classifier.pkl`
- `exports/models/nlp_semantic_config.pkl`
- `exports/models/nlp_encoder.pkl`
- `exports/plots/cm_nlp_sentence_bert.png`

## Notebooks recomendados

- `notebooks/04_nlp_text_analysis.ipynb`: baseline TF-IDF + XGBoost.
- `notebooks/05_nlp_text_analysis.ipynb`: experimento principal com Sentence-BERT local.

O notebook 05 e o mais importante para justificar a melhoria de tokenizacao/representacao textual, pois troca uma representacao baseada apenas em palavras por embeddings semanticos.

## Metricas usadas

O projeto evita usar acuracia como criterio principal. As metricas reportadas sao:

- precision;
- recall;
- F1-score;
- F1-macro;
- matriz de confusao;
- ROC-AUC apenas como metrica secundaria quando aplicavel.

F1-macro e importante neste contexto porque calcula o desempenho medio dando peso igual para cada classe, reduzindo o risco de esconder desempenho ruim em classes menores.

## Observacoes

- Nao e necessario configurar token do Hugging Face para baixar o modelo publico usado neste projeto.
- Pode aparecer um aviso sobre requisicoes nao autenticadas no Hugging Face; isso nao impede a execucao.
- O tempo de execucao do NLP pode ser maior na primeira rodada por causa do download do modelo e da geracao dos embeddings.
- O cache de embeddings pode ser apagado com seguranca; ele sera recriado na proxima execucao.
