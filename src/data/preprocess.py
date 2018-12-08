import pandas as pd

def read_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=['surname'])
    return df