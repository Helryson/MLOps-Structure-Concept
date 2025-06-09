from sklearn.model_selection import train_test_split
import joblib

def split_text(X, y, test_size=0.2):
    """
    Divide os dados em conjuntos de treino e teste.

    Parâmetros:
    X : array-like ou DataFrame
        Dados de entrada (features).
    y : array-like
        Labels ou target.
    test_size : float, opcional (default=0.2)
        Proporção do conjunto de teste.

    Retorna:
    X_train, X_test : subconjuntos de X para treino e teste.
    
    Além disso, salva y_train e y_test em arquivos pickle na pasta 'data/processed/'.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    joblib.dump(y_train, 'data/processed/y_train.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')

    return X_train, X_test
