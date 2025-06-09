"""
Este script realiza uma busca por hiperparâmetros (número de épocas e taxa de aprendizado)
usando validação cruzada (K-Fold) em um modelo de rede neural com Keras.

Fluxo principal:
- Carrega os dados de treino vetorizados.
- Converte os rótulos para one-hot encoding, se necessário.
- Define uma função para criar o modelo.
- Realiza validação cruzada com diferentes combinações de hiperparâmetros.
- Imprime os melhores parâmetros e a melhor acurácia média.
"""

import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from dataloader import load_train_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Carrega os dados de treino
X_train_vec, y_train = load_train_data()

# Converte os rótulos para one-hot se estiverem em formato categórico
if y_train.ndim == 1 or y_train.shape[1] == 1:
    y_train = to_categorical(y_train)

# Função para criar o modelo com taxa de aprendizado variável
def create_model(X_train_vec, y_train, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train_vec.shape[1],)))
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hiperparâmetros a serem testados
epochs_list = [10, 20]
learning_rates = [0.001, 0.01]
kfold = KFold(n_splits=3, shuffle=True, random_state=42)

# Inicializa variáveis para acompanhar os melhores resultados
best_score = 0
best_params = {}

# Loop por combinações de épocas e taxas de aprendizado
for epochs in epochs_list:
    for lr in learning_rates:
        scores = []
        for train_idx, val_idx in kfold.split(X_train_vec):
            X_train, X_val = X_train_vec[train_idx], X_train_vec[val_idx]
            y_train_split, y_val = y_train[train_idx], y_train[val_idx]

            model = create_model(X_train_vec, y_train, learning_rate=lr)
            model.fit(X_train, y_train_split, epochs=epochs, verbose=0)
            score = model.evaluate(X_val, y_val, verbose=0)[1]  # accuracy
            scores.append(score)

        avg_score = np.mean(scores)
        print(f"Epochs: {epochs}, LR: {lr}, Accuracy: {avg_score:.4f}")

        if avg_score > best_score:
            best_score = avg_score
            best_params = {'epochs': epochs, 'learning_rate': lr}

print("Melhores parâmetros:", best_params)
print("Melhor acurácia média:", best_score)
