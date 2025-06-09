from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def extract_features(X_train, X_test):
    """
    Transforma os dados de texto em vetores numéricos usando TF-IDF.
    Limita o número de features para evitar sobrecarga de memória.
    Salva os vetores transformados em arquivos .pkl.
    """

    # Cria o vetorizar com no máximo 5000 features para reduzir o custo computacional
    vectorizer = TfidfVectorizer(max_features=5000)

    # Ajusta o vetorizar nos dados de treino e transforma
    X_train_vec = vectorizer.fit_transform(X_train)

    # Transforma os dados de teste com o mesmo vetorizar
    X_test_vec = vectorizer.transform(X_test)

    # Salva os vetores processados
    joblib.dump(X_train_vec, 'data/processed/X_train_vec.pkl')
    joblib.dump(X_test_vec, 'data/processed/X_test_vec.pkl')
