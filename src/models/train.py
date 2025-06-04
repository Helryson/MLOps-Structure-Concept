from pathlib import Path

def train_model(model, X_train_vec, y_train, output_model_path, epochs = 10):
    """Treina o modelo usando dados processados do pipeline de dados."""

    model.fit(X_train_vec.toarray(), y_train, epochs=epochs, verbose=1)
    output_model = Path(output_model_path) / 'trained_model.h5'
    model.save(output_model)

    return output_model

    # acc = model.evaluate_model(clf, X_test_vec, y_test)
    # logger.info(f"Acurácia final: {acc:.4f}")