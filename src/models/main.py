# main.py

import logging
import click
from tensorflow.keras.models import load_model # type: ignore
from sklearn.metrics import accuracy_score
import numpy as np
import mlflow
import mlflow.keras
from mlflow.models.signature import infer_signature

from src.models import (
    load_train_data,
    load_test_data,
    create_model,
    train_model,
    predict_model
)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@click.command()
@click.argument('output_model_dir', type=click.Path(exists=True))
def main(output_model_dir):
    """Executa pipeline de processamento dos dados."""

    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    logger.info("Carregando os dados de treino...")
    X_train_vec, y_train = load_train_data()

    logger.info("Carregando dados de teste...")
    X_test_vec, y_test = load_test_data()

    logger.info("Criando modelo...")
    model = create_model(X_train_vec, y_train)

    logger.info("Executando treinamento...")
    output_model_path = train_model(model, X_train_vec, y_train, output_model_dir)
    logger.info(f'Modelo treinado salvo em {output_model_path}')

    trained_model = load_model(output_model_path)

    logger.info("Avaliando modelo...")
    y_pred = predict_model(trained_model, X_test_vec)
    y_test_labels = np.argmax(y_test, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_test_labels, y_pred_labels)

    print(f'Accuracy: {acc}')
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("teste")

    with mlflow.start_run():

        input_example = X_train_vec[:5]
        signature = infer_signature(input_example, trained_model.predict(input_example))

        mlflow.log_param("epochs", 10)
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("learning_rate", 0.01)
        mlflow.log_param("activation_hidden", "relu")
        mlflow.log_param("activation_output", "softmax")
        mlflow.log_param('loss', 'categorical_crossentropy')
        mlflow.log_metric("accuracy", acc)

        mlflow.keras.log_model(trained_model, "model", signature=signature)

    logger.info("Métricas do modelo disponível em http://localhost:5000")


if __name__ == '__main__':
    main()
