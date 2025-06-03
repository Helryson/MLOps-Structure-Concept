from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

def create_model(X_train_vec, y_train, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train_vec.shape[1],)))  # camada oculta
    model.add(Dense(y_train.shape[1], activation='softmax'))  # camada de saída

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model