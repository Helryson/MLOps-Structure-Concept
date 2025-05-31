from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def extract_features(df):

    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(df['cleaned_text']), vectorizer