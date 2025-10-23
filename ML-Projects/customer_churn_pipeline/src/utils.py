
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier

def load_data(path):
    return pd.read_csv(path)

def build_pipeline(cat_cols, num_cols):
    cat_tf = OneHotEncoder(handle_unknown="ignore")
    pre = ColumnTransformer(
        [("cat", cat_tf, cat_cols)],
        remainder="passthrough"
    )
    clf = GradientBoostingClassifier()
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    return pipe

def evaluate(model, X_val, y_val):
    pred = model.predict(X_val)
    acc = accuracy_score(y_val, pred)
    proba = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, proba)
    return {"accuracy": acc, "roc_auc": auc}
