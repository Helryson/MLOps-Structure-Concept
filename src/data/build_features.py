from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(cleaned_text_train):

    #Definindo max features para que não pese a memoria ram. Treinamento de modelos pesam mais conforme o número de colunas(features)
    vectorizer = TfidfVectorizer(max_features=5000)
    return vectorizer.fit_transform(cleaned_text_train), vectorizer