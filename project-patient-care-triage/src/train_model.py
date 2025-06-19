from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from joblib import dump

def train_rf(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print(classification_report(y_test, preds))

def save_model(model, path='model/rf_model.joblib'):
    dump(model, path)
