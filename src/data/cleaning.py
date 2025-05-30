from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import spacy

nltk.download('punkt_tab')
nltk.download('stopwords')

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = word_tokenize(texto)
    tokens_filtrados = [token for token in tokens if token not in stopwords.words('english')]
    tokens_lematizados = lematizar(tokens_filtrados)
    return tokens_lematizados

def lematizar(tokens):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(' '. join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return lemmas