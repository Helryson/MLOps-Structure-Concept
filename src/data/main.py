# main.py

import logging
import click

from src.data import (
    save_load_data,
    limpar_texto,
    map_label,
    encode,
    split_text,
    extract_features
)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@click.command()
@click.argument('output_raw_path', type=click.Path(exists=True))
def main(output_raw_path):
    """Executa pipeline de processamento dos dados."""

    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    logger.info("Carregando os dados...")
    df = save_load_data(output_raw_path, logger)
    logger.info(f'Raw dataset salvo com sucesso em {output_raw_path}')

    logger.info("Limpando e tokenizando textos...")
    df = limpar_texto(df)

    logger.info("Mapeando os rótulos...")
    df = map_label(df)

    X = df['cleaned_text']
    y = encode(df)

    logger.info("Dividindo os dados em treino e teste...")
    X_train, X_test = split_text(X, y, test_size=0.3)

    logger.info("Extraindo features do texto...")
    extract_features(X_train, X_test)

    logger.info("Pipeline executada com sucesso!")

if __name__ == '__main__':
    main()
