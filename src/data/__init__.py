import click
import logging
from pathlib import Path # Recomendado ao ter que rodar esse mesmo código em diferentes OS
import pandas as pd
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
import cleaning
import build_features
import labeling, splitting

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):

    logger = logging.getLogger(__name__)

    df = pd.read_csv(input_filepath)

    df[['cleaned_text', 'tokens_lematizados', 'tokens']] = df['text'].apply(cleaning.limpar_texto).apply(pd.Series)
    df = labeling.map_label(df)
    X = df['cleaned_text']
    y = labeling.encode(df)

    X_train, X_teste, y_train, y_teste = splitting.split_text(X, y, test_size=0.3)

    X_train_vec, vectorizer = build_features.extract_features(X_train)
    X_teste_vec = vectorizer.transform(X_teste)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Retorna a raiz do pro
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
