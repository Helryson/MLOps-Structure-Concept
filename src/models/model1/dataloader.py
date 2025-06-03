import joblib

def load_train_data():
    X_train_vec = joblib.load('data/processed/X_train_vec.pkl')
    y_train = joblib.load('data/processed/y_train.pkl')

    return X_train_vec, y_train

def load_test_data():
    X_test_vec = joblib.load('data/processed/X_test_vec.pkl')
    y_test = joblib.load('data/processed/y_test.pkl')

    return X_test_vec, y_test