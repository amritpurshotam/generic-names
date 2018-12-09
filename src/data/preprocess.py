import pandas as pd
from numpy import ndarray
from sklearn.feature_extraction.text import CountVectorizer
from typing import Tuple

def read_raw_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df.drop(columns=['surname'], inplace=True)
    df = strip_whitespace(df)
    df = drop_names_with_different_labels(df)
    df = drop_duplicated_rows(df)
    return df

def vectorize(raw_text: ndarray) -> Tuple[ndarray, CountVectorizer]:
    vectorizer = CountVectorizer(lowercase=False, analyzer='char', strip_accents='ascii')
    feature_vectors = vectorizer.fit_transform(raw_text).toarray()
    return feature_vectors, vectorizer

def drop_names_with_different_labels(df: pd.DataFrame) -> pd.DataFrame:
    df['duplicates_name'] = df.duplicated(subset=['name'])
    df['duplicates_name_and_label'] = df.duplicated(subset=['name', 'name_generic'])
    names = df[df.duplicates_name != df.duplicates_name_and_label].name
    indexes_to_drop = df[df.name.isin(names)].index.values
    df.drop(labels=indexes_to_drop, inplace=True)
    df.drop(columns=['duplicates_name', 'duplicates_name_and_label'], inplace=True)
    return df

def drop_duplicated_rows(df: pd.DataFrame) -> pd.DataFrame:
    df.drop_duplicates(subset=['name', 'name_generic'], inplace=True)
    return df

def strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    df.name = df.name.str.strip()
    return df