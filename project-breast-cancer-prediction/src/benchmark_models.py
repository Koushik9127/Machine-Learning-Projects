from lazypredict.Supervised import LazyClassifier
from sklearn.model_selection import train_test_split

def benchmark(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LazyClassifier()
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    print(models)
