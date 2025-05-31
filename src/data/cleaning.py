from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re
import spacy

nltk.download('punkt_tab')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm')

def limpar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = word_tokenize(texto)
    tokens_filtrados = [token for token in tokens if token not in stopwords.words('english')]
    tokens_lematizados = lematizar(tokens_filtrados)
    return ' '.join(tokens_lematizados), tokens_lematizados, tokens

def lematizar(tokens):
    doc = nlp(' '. join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return lemmas