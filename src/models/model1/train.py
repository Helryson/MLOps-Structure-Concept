import click
import logging
from dotenv import find_dotenv, load_dotenv

from src.data import load_and_process_data  # ou use `from data.pipeline import load_and_process_data`
from src.models.model1 import model  # model.py com funções de treinamento e avaliação

@click.command()
@click.argument('output_raw_path', type=click.Path(exists=True))
def train(output_raw_path):
    """Treina o modelo usando dados processados do pipeline de dados."""

    # Setup de logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    # Carregamento e pré-processamento dos dados
    logger.info("Iniciando o pipeline de dados...")
    X_train_vec, X_test_vec, y_train, y_test = load_and_process_data(output_raw_path, logger)

    # Treinamento do modelo
    logger.info("Treinando o modelo...")
    clf = model.train_model(X_train_vec, y_train)

    # Avaliação
    logger.info("Avaliando o modelo...")
    acc = model.evaluate_model(clf, X_test_vec, y_test)
    logger.info(f"Acurácia final: {acc:.4f}")

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    train()
