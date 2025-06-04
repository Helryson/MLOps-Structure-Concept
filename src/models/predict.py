import numpy as np

def predict_model(model, X_test_vec):
    y_pred = model.predict(X_test_vec)
    return y_pred