import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('outtu_filepath', type=click.Path())
def main(input_filepath, output_filepath):

    logger = logging.getLogger(__name__)

    logger.info('Carregando raw dataset')
    df = pd.read_csv(input_filepath)

    nltk.download('punkt_tab')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    df['tokens'] = df['text'].apply(word_tokenize)
    df['cleaned_tokens'] = df['tokens'].apply(lambda tokens: [word for word in tokens if word.lower() not in stop_words])