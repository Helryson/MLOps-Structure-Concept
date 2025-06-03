from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(X_train, X_test):

    #Definindo max features para que não pese a memoria ram. Treinamento de modelos pesam mais conforme o número de colunas(features)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    joblib.dump(X_train_vec, 'data/processed/X_train_vec.pkl')
    joblib.dump(X_test_vec, 'data/processed/X_test_vec.pkl')