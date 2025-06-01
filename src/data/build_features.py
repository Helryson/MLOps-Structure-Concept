from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(cleaned_text_train):

    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(cleaned_text_train), vectorizer