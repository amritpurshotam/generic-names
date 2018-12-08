from src.data import read_raw_data, preprocess

raw_data_path = r'data/raw/A_training data.csv'

def test_raw_shape():
    df = read_raw_data(raw_data_path)
    assert df.shape == (4322, 3)

def test_name_equals_surname():
    df = read_raw_data(raw_data_path)
    assert df.name.equals(df.surname) == True

def test_preprocess():
    df = read_raw_data(raw_data_path)
    df = preprocess(df)
    assert df.columns.tolist() == ['name', 'name_generic']