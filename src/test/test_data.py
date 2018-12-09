from src.data import *
from pandas import DataFrame
from numpy import array

raw_data_path = r'data/raw/A_training data.csv'

def test_raw_shape():
    df = read_raw_data(raw_data_path)
    assert df.shape == (4322, 3)

def test_name_equals_surname():
    df = read_raw_data(raw_data_path)
    assert df.name.equals(df.surname) == True

def test_correct_columns_are_selected():
    df = DataFrame(columns=['name', 'surname', 'name_generic'])
    df = clean(df)
    assert df.columns.tolist() == ['name', 'name_generic']

def test_mislabeled_rows_dropped():
    df = DataFrame(data={'name': ['Global', 'Global', 'Amrit'], 'name_generic': [0, 1, 0]})
    df = drop_names_with_different_labels(df)
    assert df.iloc[0]['name'] == 'Amrit'
    assert df.shape == (1, 2)

def test_drop_duplicated_rows():
    df = DataFrame(data={'name': ['Amrit', 'Amrit'], 'name_generic': [0, 0]})
    df = drop_duplicated_rows(df)
    assert df.shape == (1, 2)

def test_strip_whitespace():
    df = DataFrame(data={'name': [' Amrit ']})
    df = strip_whitespace(df)
    assert df.iloc[0]['name'] == 'Amrit'

def test_vectorize():
    raw_docs = ['Abé', 'cd']
    features, vectorizer = vectorize(raw_docs)
    expected = array([[1, 1, 0, 0, 1], [0, 0, 1, 1, 0]], dtype='int64')
    assert (features == expected).all() == True
    assert vectorizer.get_feature_names() == ['A', 'b', 'c', 'd', 'e']