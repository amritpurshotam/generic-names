import pandas as pd

def read_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['surname'], inplace=True)
    df = drop_names_with_different_labels(df)
    return df

def drop_names_with_different_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['duplicates_name'] = df.duplicated(subset=['name'])
    df['duplicates_name_and_label'] = df.duplicated(subset=['name', 'name_generic'])
    names = df[df.duplicates_name != df.duplicates_name_and_label].name
    indexes_to_drop = df[df.name.isin(names)].index.values
    df.drop(labels=indexes_to_drop, inplace=True)
    df.drop(columns=['duplicates_name', 'duplicates_name_and_label'], inplace=True)
    return df