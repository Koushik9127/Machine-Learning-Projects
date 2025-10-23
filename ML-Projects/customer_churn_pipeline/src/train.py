
import joblib, os, mlflow, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_data, build_pipeline, evaluate

DATA_PATH = "data/churn.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    if not os.path.exists(DATA_PATH):
        sys.exit("✖ Dataset not found. Place CSV at data/churn.csv")
    df = load_data(DATA_PATH)
    target = "Churn"
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = build_pipeline(cat_cols, num_cols)

    mlflow.set_experiment("customer-churn")
    with mlflow.start_run():
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_val, y_val)
        mlflow.log_metrics(metrics)
        model_path = os.path.join(MODEL_DIR, "model.pkl")
        joblib.dump(pipe, model_path)
        mlflow.log_artifact(model_path)
        print("✅ Training complete:", metrics)

if __name__ == "__main__":
    main()
