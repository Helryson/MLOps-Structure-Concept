# MLOps Structure Concept

Este repositório apresenta uma estrutura modular e escalável para projetos de **MLOps** com Python, organizando as etapas de ingestão de dados, treinamento de modelos, e visualização de resultados.

A estrutura é pensada para facilitar a manutenção, a reprodutibilidade e a implementação de boas práticas no ciclo de vida de Machine Learning.

---

## Estrutura de Pastas

```
MLOps-Structure-Concept/
├── data/ # Armazena os dados brutos e processados
│ └── raw/
├── models/ # Modelos treinados e artefatos
├── notebooks/ # Jupyter notebooks para exploração
├── src/ # Código-fonte principal
│ ├── data/ # Pré-processamento de dados
│ ├── models/ # Treinamento e avaliação de modelos
│ └── visualization/ # Visualização dos resultados
├── requirements.txt # Dependências do projeto
└── README.md # Documentação
```
---

## Pré-requisitos

Antes de executar os módulos, crie e ative um ambiente virtual (recomendado) e instale as dependências:

```
pip install -r requirements.txt
```


## Execução dos Módulos
A execução dos módulos pode ser feita diretamente da raiz do projeto.

#### Todos os comandos devem ser executados a partir da raiz (MLOps-Structure-Concept/)
Certifique-se de que os diretórios src/ e subpastas contenham arquivos __init__.py.

### 1. Processamento de Dados
```
python3 -m src.data.main data/raw/
```
Função: Realiza o carregamento e processamento inicial dos dados.

Argumento: Caminho para o diretório contendo os dados brutos.

### 2. Treinamento de Modelos
```
python3 -m src.models.main models/
```
Função: Treina os modelos e salva os artefatos no diretório indicado.

Argumento: Caminho onde os modelos treinados serão armazenados.

### 3. Visualização de Resultados
```
python3 -m src.visualization.evaluation
```
Função: Gera visualizações e métricas dos modelos previamente treinados.

Argumento: Nenhum.
