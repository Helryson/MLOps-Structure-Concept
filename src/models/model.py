from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import Input # type: ignore

def create_model(X_train_vec, y_train, learning_rate=0.01):
    model = Sequential([
        Input(shape=(X_train_vec.shape[1],)),
        Dense(10, activation='relu'),
        Dense(y_train.shape[1], activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model