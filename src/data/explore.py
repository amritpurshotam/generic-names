from preprocess import read_raw_data, preprocess
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

def count_characters(df: pd.DataFrame) -> pd.DataFrame:
    count_dict = Counter(df.name.str.cat())
    count_df = pd.DataFrame.from_dict(count_dict, orient='index')
    count_df.reset_index(inplace=True)
    count_df.rename(columns={'index': 'Character', 0: 'Count'}, inplace=True)
    count_df.Count = count_df.Count.astype(float)
    size = df.shape[0]
    count_df.Count = count_df.Count / size
    count_df.sort_values(by='Count', inplace=True, ascending=False)
    return count_df

if __name__ == "__main__":
    df = read_raw_data('data\\raw\\A_training data.csv')
    df = preprocess(df)

    generic = df[df.name_generic == 1]
    non_generic = df[df.name_generic == 0]

    count_generic = count_characters(generic)
    count_non_generic = count_characters(non_generic)

    count_df = count_non_generic.merge(right=count_generic, how='left', on='Character')
    count_df.fillna(0, inplace=True)
    count_df.rename(columns={'Count_x': 'Count_Non_Generic', 'Count_y': 'Count_Generic'}, inplace=True)
    count_df.plot(x='Character', y=['Count_Non_Generic', 'Count_Generic'], kind='bar')
    plt.show()