from pathlib import Path

def train_model(model, X_train_vec, y_train, output_model_path, epochs = 10):
    """
    Treina e salva o modelo.

    Parâmetros:
    - model: modelo Keras não treinado.
    - X_train_vec: dados de treino vetorizados.
    - y_train: rótulos de treino.
    - output_model_path: caminho da pasta para salvar o modelo.
    - epochs: número de épocas (padrão: 10).

    Retorna:
    - Caminho do modelo salvo (.keras).
    """

    model.fit(X_train_vec.toarray(), y_train, epochs=epochs, verbose=1)
    output_model = Path(output_model_path) / 'trained_model.keras'
    model.save(output_model)

    return output_model