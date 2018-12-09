import sys
sys.path.append('src')
from src.modelling import XgbModel
from pandas import DataFrame

if __name__ == "__main__":
    model = XgbModel()
    model.load('models')

    df = DataFrame(data={'Name': ['Amrit', 'ServerRoom'], 'Surname': ['Purshotam', 'Amrit']})

    df['Is_name_generic'] = model.predict(df.Name)
    df['Is_surname_generic'] = model.predict(df.Surname)
    df['Is_record_generic'] = 0
    df.loc[(df.Is_name_generic == 1) & (df.Is_surname_generic == 1), 'Is_record_generic'] = 1

    df.to_csv(path_or_buf='reports/output.csv', index=False)