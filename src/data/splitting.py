from sklearn.model_selection import train_test_split
import joblib

def split_text(X, y, test_size = 0.2):

    '''Separa o dataframe '''

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    joblib.dump(y_train, 'data/processed/y_train.pkl')
    joblib.dump(y_test, 'data/processed/y_test.pkl')

    return X_train, X_test