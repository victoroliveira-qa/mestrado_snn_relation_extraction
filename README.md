# SNN Research — Etapa 3 (modelos de embeddings configuráveis e artefatos organizados)

Este repositório refatora o notebook **`Evolução_SNN_Versão3.ipynb`**, preservando o MLP compartilhado da rede Siamese, a fusão ordenada da Versão 3, a camada de decisão binária para pares e a inferência multiclasse **separada**, baseada em protótipos. O código-fonte e seus comentários permanecem em inglês.

## Estrutura do projeto

- `main.py`: interface de linha de comando (CLI) para treinamento e inferência.
- `configs/experiments/`: configurações independentes dos experimentos.
- `datasets/raw/`: datasets originais CSV/XLSX (não incluídos no pacote).
- `datasets/embeddings/`: embeddings FAISS/NPY pré-computados (não incluídos no pacote).
- `src/data/`: carregamento, divisão dos dados e geração de pares.
- `src/embeddings/providers.py`: provedores intercambiáveis de embeddings FAISS e NPY **pré-computados**.
- `src/models/siamese.py`: MLP de projeção compartilhado e MLP de classificação de pares da rede Siamese.
- `src/evaluation/`: protótipos, métricas e gráficos.
- `src/inference/`: predição de uma relação a partir de um vetor.
- `experiments/`: pastas de resultados criadas durante a execução.
- `tests/`: verificações automatizadas iniciais.

## Preparação do ambiente

Recomenda-se Python 3.10 a 3.12. O uso de GPU é opcional para execuções pequenas. Em um ambiente virtual novo, execute:

```bash
pip install -r requirements.txt
```

Coloque os arquivos originais nos caminhos indicados em `configs/experiments/v3_qwen_business.yaml`:

- `datasets/raw/DATASET_TREINO_BASE_merged.xlsx`
- `datasets/embeddings/qwen_all_in_one_semantics_index.faiss`

**CRÍTICO:** a Versão 3 pressupõe que a linha 0 da planilha original corresponde ao vetor de ID 0 no FAISS, e assim sucessivamente. Essa associação **não está armazenada no próprio vetor FAISS**. Verifique o script que gerou os embeddings, a ordem dos registros de origem e eventuais exclusões de linhas. **Somente depois de confirmar o alinhamento**, defina `embedding.positional_alignment_verified: true` no YAML. Se a verificação falhar, será necessário um manifesto com identificadores estáveis e um carregador diferente; nunca altere essa configuração apenas para contornar a verificação.

## Treinamento

```bash
python main.py train --config configs/experiments/v3_qwen_business.yaml
```

Os resultados serão salvos em `experiments/v3_qwen_business/`. As pastas de experimentos existentes não podem ser sobrescritas: altere `experiment.name` para iniciar uma nova execução.

## Inferência (registro do dataset original)

```bash
python main.py infer --config configs/experiments/v3_qwen_business.yaml --experiment-dir experiments/v3_qwen_business --row-id 0
```

A saída multiclasse é ordenada pela **similaridade de cosseno**, e não pela probabilidade de classe. A inferência sobre pares constitui uma tarefa distinta.

## Troca de embeddings ou datasets

Copie o arquivo YAML de configuração e altere `experiment.name`, `dataset.path`, `dataset.label_column`, `embedding.provider` e `embedding.path`. O provedor `npy` aceita uma matriz bidimensional de valores de ponto flutuante, com um vetor por linha do dataset original; a dimensão é identificada automaticamente.

É possível adicionar um adaptador de geração para Qwen/BGE/E5 sem modificar a implementação da Siamese. **Observação histórica:** essa funcionalidade ainda não estava implementada na etapa anterior; a geração direta desses embeddings foi acrescentada na Etapa 3, conforme descrito mais adiante. O provedor FAISS recupera embeddings **já existentes** do Qwen; não executa inferência no modelo Qwen.

## Cuidados metodológicos da pesquisa

- O alinhamento posicional entre entradas e FAISS precisa ser verificado. A Versão 3 não contém identificadores estáveis.
- A divisão 80/10/10 é feita **por linha**, não por entidade. Ainda existe risco de vazamento de dados entre linhas relacionadas à mesma entidade; a avaliação com divisão por grupos deverá constituir um experimento separado.
- O padrão para a divisão é `stratify: auto`, que tenta estratificar e, caso não seja possível, recorre à divisão aleatória. Para exigir uma divisão estritamente estratificada, utilize `stratify: strict`; classes raras podem provocar um erro explícito.
- A validação utiliza pares fixos e reproduzíveis, sem acessar amostras de teste. Os geradores de pares produzem um erro explícito quando um subconjunto não permite formar pares positivos e negativos.
- A fusão padrão da Versão 3 `[left, right, absdiff, product]` é sensível à ordem dos exemplos; isso foi mantido para preservar o notebook original. Existe uma alternativa simétrica configurável, mas ela deverá ser documentada como um experimento diferente.
- A classificação multiclasse usa similaridade de cosseno em relação a **protótipos calculados apenas com o treino**; a classificação binária de pares utiliza uma camada sigmoid treinada. São avaliações distintas.
- Não são utilizados protótipos zerados para classes ausentes do conjunto de treino; as predições excluem essas classes, mas as métricas informam todas as classes conhecidas.
- Diferenças entre dependências de CPU/GPU e operações não determinísticas podem afetar a reprodutibilidade numérica exata.
- A refatoração do código, isoladamente, não comprova reprodução validada: os dados e o FAISS originais não estão incluídos neste pacote. Caso você já tenha validado a Etapa 2 com seus arquivos, preserve os resultados dessa validação.

## Testes automatizados

```bash
python -m pytest -q
```

## Etapa 3: geração de embeddings com modelos configuráveis

Instale os pacotes opcionais para inferência com modelos após instalar as dependências principais:

```bash
pip install -r requirements-embeddings.txt
```

Utilize um YAML experimental com `embedding.provider: model` e defina `embedding.model` como uma das opções:

- `qwen3-embedding-4b`
- `qwen3-embedding-8b`
- `bge-m3`
- `multilingual-e5-large`

Como alternativa, informe explicitamente `embedding.model_name` com o identificador de um modelo do Hugging Face. Os exemplos estão em `configs/experiments/*_generated.yaml`.

**Antes de executar**, ajuste `embedding.text_columns`, `separator`, `text_prefix` e eventuais prompts do modelo para refletir as particularidades do dataset e a metodologia anteriormente utilizada para gerar os embeddings. Os nomes das colunas nos exemplos são ilustrativos: não foram verificados contra o esquema real do Business. Os exemplos com E5 usam o prefixo `passage: `.

A geração de embeddings diretamente pelos modelos **não garante equivalência** com os vetores FAISS usados na Versão 3 se houver diferenças na construção dos textos, prompts ou truncamento. Os modelos Qwen de 4B e 8B geralmente exigem uma GPU configurada adequadamente e com memória suficiente. O uso exclusivo de CPU pode ser impraticável.

Os vetores gerados são armazenados em cache como uma matriz NPY bidimensional em `datasets/embeddings/generated/`, acompanhada de um manifesto JSON. A chave do cache considera modelo, configuração de revisão, conteúdo textual e opções de formação dos textos. O cache só será reutilizado quando metadados e dimensões corresponderem. Não modifique manualmente arquivos NPY gerados.

## Etapa 3: organização dos resultados

Cada execução experimental passa a salvar:

```text
experiments/<experiment_name>/
  configs/config.yaml
  models/{siamese.keras,encoder.keras,prototypes.npy,label_classes.npy,missing_train_classes.json}
  splits/split_indices.npz
  metrics/{validation_metrics.json,test_metrics.json}
  predictions/predictions.csv
  figures/{loss.png,accuracy.png,auc.png,confusion_matrix.png,tsne.png,pacmap.png}
  logs/{training_history.csv,metadata.json}
```

Somente os caminhos de saída foram reorganizados: o modelo Siamese de referência, o gerador de pares, os hiperparâmetros de treinamento e o código de avaliação permanecem inalterados. As pastas de experimentos criadas na Etapa 2 continuam compatíveis com a versão antiga do código. Esta versão utiliza a estrutura da Etapa 3 para inferência; portanto, execute cada resultado com a versão correspondente do projeto. Os experimentos já concluídos não serão migrados nem reescritos automaticamente.
