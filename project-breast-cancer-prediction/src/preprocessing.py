import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_breast_cancer_data(path):
    df = pd.read_csv(path)
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis'])
    X = df.drop(['id', 'diagnosis'], axis=1)
    y = df['diagnosis']
    X_scaled = StandardScaler().fit_transform(X)
    return X_scaled, y
