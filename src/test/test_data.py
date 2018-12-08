from src.data import read_raw_data

def test_raw_shape():
    df = read_raw_data(r'data/raw/A_training data.csv')
    assert df.shape == (4322, 3)