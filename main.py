import os
from src.modelling import XgbModel
from pandas import DataFrame
import pandas as pd

if __name__ == "__main__":
    model = XgbModel()
    model.load('models')

    file_path = input('Please enter the path to the input csv file: ')
    try:
        df = pd.read_csv(file_path)

        df['Is_name_generic'] = model.predict(df.Name)
        df['Is_surname_generic'] = model.predict(df.Surname)
        df['Is_record_generic'] = 0
        df.loc[(df.Is_name_generic == 1) & (df.Is_surname_generic == 1), 'Is_record_generic'] = 1

        abs_path = os.path.abspath(file_path)
        base_dir = os.path.dirname(abs_path)
        df.to_csv(path_or_buf=base_dir + '/output.csv', index=False)
    except FileNotFoundError as e:
        print('Invalid path. Please try again.')

    