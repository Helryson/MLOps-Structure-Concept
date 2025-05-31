import pandas as pd
from src.data import build_features, labeling, splitting, cleaning, ingestion

def load_and_process_data(output_raw_path, logger):
    """
    Carrega, limpa, rotula, divide e extrai features dos dados.

    Args:
        output_raw_path (str or Path): caminho da pasta com dados brutos.
        logger (logging.Logger): logger para mensagens.

    Returns:
        X_train_vec, X_test_vec, y_train, y_test, vectorizer
    """

    logger.info("Carregando os dados...")
    df = ingestion.save_load_data(output_raw_path, logger)

    logger.info("Limpando e tokenizando textos...")
    df = cleaning.limpar_texto(df)

    logger.info("Mapeando os rótulos...")
    df = labeling.map_label(df)

    X = df['cleaned_text']
    y = labeling.encode(df)

    logger.info("Dividindo os dados em treino e teste...")
    X_train, X_test, y_train, y_test = splitting.split_text(X, y, test_size=0.3)

    logger.info("Extraindo features do texto...")
    X_train_vec, vectorizer = build_features.extract_features(X_train)
    X_test_vec = vectorizer.transform(X_test)

    logger.info("Dados prontos para treinamento.")
    return X_train_vec, X_test_vec, y_train, y_test, vectorizer
