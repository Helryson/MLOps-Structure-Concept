def evaluate_model(model, X_test_vec, y_test):
    acc = model.evaluate(X_test_vec.toarray(), y_test, verbose=0)[1]
    return acc