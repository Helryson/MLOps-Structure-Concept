from src.models.model1 import model, dataloader  # model.py com funções de treinamento e avaliação

X_train_vec, y_train = dataloader.load_train_data()
model = model.create_model(X_train_vec, y_train, epochs=25)

def train(epochs):
    """Treina o modelo usando dados processados do pipeline de dados."""

    model.fit(X_train_vec.toarray(), y_train, epochs=epochs, verbose=1)

    model.train_model(X_train_vec, y_train)
    model.save("models/modelo_1.h5")

    # acc = model.evaluate_model(clf, X_test_vec, y_test)
    # logger.info(f"Acurácia final: {acc:.4f}")