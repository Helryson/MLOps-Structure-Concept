from pathlib import Path # Recomendado ao ter que rodar esse mesmo código em diferentes OS
import pandas as pd
from datasets import load_dataset

def save_load_data(output_raw_path, logger):

    """
    Carrega o dataset 'ag_news' do Hugging Face, salva como CSV no caminho indicado e retorna o DataFrame concatenado.

    Parâmetros:
    - output_raw_path (str ou Path): diretório onde o CSV será salvo
    - logger (logging.Logger): logger para mensagens de log

    Retorna:
    - DataFrame com todos os dados unidos
    """
    dataset = load_dataset('ag_news')

    # Junta os splits train/test em um único DataFrame
    full_df = pd.concat([pd.DataFrame(dataset[split]) for split in dataset.keys()], ignore_index=True)

    output_file = Path(output_raw_path) / 'raw_data.csv'
    logger.info(f'Dataset carregado. Salvando dados em: {output_file}')
    full_df.to_csv(output_file, encoding='utf-8', index=False)

    return full_df