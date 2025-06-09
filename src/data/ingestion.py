from pathlib import Path  # Facilita caminhos multiplataforma
import pandas as pd
from datasets import load_dataset

def save_load_data(output_raw_path, logger):
    """
    Carrega o dataset 'ag_news' do Hugging Face, salva como CSV e retorna como DataFrame único.

    Parâmetros:
    - output_raw_path (str ou Path): Caminho para salvar o arquivo CSV.
    - logger (logging.Logger): Objeto logger para registrar mensagens.

    Retorna:
    - full_df (DataFrame): Dataset completo unificado (train + test).
    """

    # Carrega os splits do dataset ag_news
    dataset = load_dataset('ag_news')

    # Concatena os splits (train/test) em um único DataFrame
    full_df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset.keys()], ignore_index=True)

    # Define o caminho completo para salvar o arquivo
    output_file = Path(output_raw_path) / 'raw_data.csv'

    # Loga e salva o DataFrame como CSV
    logger.info(f'Dataset carregado. Salvando dados em: {output_file}')
    full_df.to_csv(output_file, encoding='utf-8', index=False)

    return full_df