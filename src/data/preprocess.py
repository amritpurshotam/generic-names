import pandas as pd

def read_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df