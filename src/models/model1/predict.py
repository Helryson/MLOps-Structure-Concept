
def evaluate_model(model, X_test_vec, y_test):
    loss, acc = model.evaluate(X_test_vec.toarray(), y_test, verbose=0)
    return acc