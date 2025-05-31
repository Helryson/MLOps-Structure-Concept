from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

def train_model(X_train_vec, y_train, epochs=20, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(X_train_vec.shape[1],)))  # camada oculta
    model.add(Dense(y_train.shape[1], activation='softmax'))  # camada de saída

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train_vec.toarray(), y_train, epochs=epochs, verbose=1)
    return model

def evaluate_model(model, X_test_vec, y_test):
    loss, acc = model.evaluate(X_test_vec.toarray(), y_test, verbose=0)
    return acc