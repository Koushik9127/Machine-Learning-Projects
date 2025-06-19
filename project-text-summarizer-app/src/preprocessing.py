import pandas as pd

def clean_text(text):
    return text.lower().strip()

def load_dataset(path):
    df = pd.read_csv(path)
    df['article'] = df['article'].apply(clean_text)
    return df
