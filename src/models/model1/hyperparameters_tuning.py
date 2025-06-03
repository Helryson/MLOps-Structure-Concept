from sklearn.model_selection import GridSearchCV
import dataloader  # model.py com funções de treinamento e avaliação
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from scikeras.wrappers import KerasClassifier

X_train_vec, y_train = dataloader.load_train_data()

param_grid = {
    'epochs': [10, 20],
    'learning_rate': [0.001, 0.01]
}

def create_model(input_dim, output_dim, learning_rate=0.01):
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(input_dim,)))  # camada oculta
    model.add(Dense(output_dim, activation='softmax'))  # camada de saída

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

clf = KerasClassifier(build_fn=create_model, verbose=0)

grid = GridSearchCV(estimator=clf, param_grid=param_grid, cv=3)
grid_result = grid.fit(X_train_vec.toarray(), y_train)

print("Melhores parâmetros:", grid_result.best_params_)
print("Melhor acurácia média:", grid_result.best_score_)