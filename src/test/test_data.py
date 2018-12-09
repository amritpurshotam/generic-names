from src.data import read_raw_data, preprocess, drop_names_with_different_labels
from pandas import DataFrame

raw_data_path = r'data/raw/A_training data.csv'

def test_raw_shape():
    df = read_raw_data(raw_data_path)
    assert df.shape == (4322, 3)

def test_name_equals_surname():
    df = read_raw_data(raw_data_path)
    assert df.name.equals(df.surname) == True

def test_correct_columns_are_selected():
    df = DataFrame(columns=['name', 'surname', 'name_generic'])
    df = preprocess(df)
    assert df.columns.tolist() == ['name', 'name_generic']

def test_mislabeled_rows_dropped():
    df = DataFrame(data={'name': ['Global', 'Global', 'Amrit'], 'name_generic': [0, 1, 0]})
    df = drop_names_with_different_labels(df)
    assert df.iloc[0]['name'] == 'Amrit'
    assert df.columns.tolist() == ['name', 'name_generic']