# main.py

import logging
import click
from tensorflow.keras.models import load_model # type: ignore

from src.models import (
    load_train_data,
    load_test_data,
    create_model,
    train_model,
    evaluate_model
)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@click.command()
@click.argument('output_model_path', type=click.Path(exists=True))
def main(output_model_path):
    """Executa pipeline de processamento dos dados."""

    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    logger.info("Carregando os dados de treino...")
    X_train_vec, y_train = load_train_data()

    logger.info("Carregando dados de teste...")
    X_test_vec, y_test = load_test_data()

    logger.info("Carregando modelo...")
    model = create_model(X_train_vec, y_train)

    logger.info("Treinando modelo...")
    output_model = train_model(model, X_train_vec, y_train, output_model_path)
    logger.info(f'Modelo salvo em {output_model}')

    trained_model = load_model(output_model)

    logger.info("Avaliando modelo...")
    score = evaluate_model(trained_model, X_test_vec, y_test)

    print(f'Accuracy: {score}')

    logger.info("Pipeline rodada com sucesso!")

if __name__ == '__main__':
    main()
