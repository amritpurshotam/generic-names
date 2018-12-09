import sys
sys.path.append('src')

from data import read_raw_data
from modelling import XgbModel

if __name__ == "__main__":
    df = read_raw_data('data\\raw\\A_training data.csv')
    model = XgbModel()
    model.train(df)
    model.save('models')