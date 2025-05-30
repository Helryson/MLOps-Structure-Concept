import click
import logging
from pathlib import Path # Recomendado ao ter que rodar esse mesmo código em diferentes OS
import pandas as pd
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv

@click.command()
@click.argument('output_filepath', type=click.Path(exists=True))
def main(output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('Carregando dataset através do huggingface')
    df = load_dataset('ag_news')

    logger.info(f'Dataset carregado. Salvando dados em {input_file}')
    # Path(output_filepath).__truediv__('raw_data.csv')
    input_file = Path(output_filepath) / 'raw_data.csv'
    full_df = pd.concat([pd.DataFrame(df[split]) for split in df], ignore_index=True)
    full_df.to_csv(input_file, index=False)
    logger.info('Raw dataset salvo.')

    # logger.info('Processando dataset')
    # train_df = pd.DataFrame(df['train'])
    # train_df = train_df[['text', 'label']]

    # output_file = Path(output_filepath) / 'train_data.csv'
    # train_df.to_csv(output_file, index=False)

    # logger.info(f'Dados processados salvos em {output_file}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Retorna a raiz do pro
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
