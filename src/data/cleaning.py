import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from tqdm import tqdm

nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

def lematizar(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def limpar_texto(df):
    stop_words = set(stopwords.words('english'))

    all_tokens = []
    all_tokens_lematizados = []
    all_cleaned_text = []

    for texto in tqdm(df['text'], desc="Limpando textos"):
        texto = texto.lower()
        texto = re.sub(r'[^\w\s]', '', texto)
        tokens = word_tokenize(texto)
        tokens_filtrados = [t for t in tokens if t not in stop_words]
        tokens_lematizados = lematizar(tokens_filtrados)
        cleaned_text = ' '.join(tokens_lematizados)

        all_tokens.append(tokens)
        all_tokens_lematizados.append(tokens_lematizados)
        all_cleaned_text.append(cleaned_text)

    df['tokens'] = all_tokens
    df['tokens_lematizados'] = all_tokens_lematizados
    df['cleaned_text'] = all_cleaned_text

    return df