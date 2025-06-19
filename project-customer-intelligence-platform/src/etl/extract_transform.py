import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    print("✅ Loaded data with shape:", df.shape)
    return df

def clean_data(df):
    df = df.dropna()
    df = df[df.select_dtypes(include='number').columns]
    print("🧹 Cleaned data")
    return df
