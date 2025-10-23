def run_model(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict(X_test)
