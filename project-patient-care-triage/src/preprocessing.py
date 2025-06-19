import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.dropna()
    X = df.drop('Triage_Level', axis=1)
    y = df['Triage_Level']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
