from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import Input # type: ignore

def create_model(X_train_vec, y_train, learning_rate=0.01):

    """
    Cria e compila um modelo Keras com:
    - 1 camada densa (ReLU) e 1 de saída (Softmax).
    - Usa 'categorical_crossentropy' e Adam como otimizador.

    Parâmetros:
    - X_train_vec: dados de entrada vetorizados.
    - y_train: rótulos em one-hot.
    - learning_rate: taxa de aprendizado.

    Retorna:
    - Modelo compilado.
    """
    model = Sequential([
        Input(shape=(X_train_vec.shape[1],)),
        Dense(10, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model