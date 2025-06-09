import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import spacy
from tqdm import tqdm

# Baixa os recursos necessários do NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Carrega o modelo de linguagem em inglês do spaCy
nlp = spacy.load('en_core_web_sm')

def lematizar(tokens):
    """
    Aplica lematização usando spaCy em uma lista de tokens.
    Retorna os lemas dos tokens.
    """
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def limpar_texto(df):
    """
    Realiza o pré-processamento dos textos em um DataFrame.
    As etapas incluem:
    - Conversão para minúsculas
    - Remoção de pontuação
    - Tokenização
    - Remoção de stopwords
    - Lematização
    - Geração do texto limpo final

    Retorna o DataFrame com novas colunas:
    - 'tokens': tokens originais
    - 'tokens_lematizados': tokens lematizados
    - 'cleaned_text': texto final limpo
    """

    stop_words = set(stopwords.words('english'))

    all_tokens = []
    all_tokens_lematizados = []
    all_cleaned_text = []

    for texto in tqdm(df['text'], desc="Limpando textos"):
        texto = texto.lower()  # converte para minúsculas
        texto = re.sub(r'[^\w\s]', '', texto)  # remove pontuação
        tokens = word_tokenize(texto)  # tokeniza o texto
        tokens_filtrados = [t for t in tokens if t not in stop_words]  # remove stopwords
        tokens_lematizados = lematizar(tokens_filtrados)  # aplica lematização
        cleaned_text = ' '.join(tokens_lematizados)  # junta os tokens em texto limpo

        all_tokens.append(tokens)
        all_tokens_lematizados.append(tokens_lematizados)
        all_cleaned_text.append(cleaned_text)

    # Adiciona as colunas ao DataFrame original
    df['tokens'] = all_tokens
    df['tokens_lematizados'] = all_tokens_lematizados
    df['cleaned_text'] = all_cleaned_text

    return df