import numpy as np

def predict_model(model, X_test_vec):
    """
    Gera previsões a partir do modelo treinado.

    Parâmetros:
    - model: modelo Keras já treinado.
    - X_test_vec: dados de teste vetorizados.

    Retorna:
    - y_pred: previsões do modelo.
    """
    
    y_pred = model.predict(X_test_vec)
    return y_pred