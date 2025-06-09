# Importa os dados de teste e a função de carregamento de modelo
from src.visualization import load_test_data
from tensorflow.keras.models import load_model  # type: ignore

# Função para avaliar o modelo com os dados de teste
def evaluate_model(model, X_test_vec, y_test):
    loss, acc = model.evaluate(X_test_vec.toarray(), y_test, verbose=0)
    return loss, acc

# Carrega os dados de teste vetorizados e os rótulos
X_test_vec, y_test = load_test_data()

# Carrega o modelo treinado salvo no disco
model = load_model('models/trained_model.keras')

# Avalia o modelo com os dados de teste
loss, acc = evaluate_model(model, X_test_vec, y_test)

# Exibe os resultados de perda e acurácia
print()
print(f'Loss: {loss}')
print(f'Accuracy: {acc}')