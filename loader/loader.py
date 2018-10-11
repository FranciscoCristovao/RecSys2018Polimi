import pandas as pd


def load_dataframe(path, sep):
    df = pd.read_csv(path, sep=sep)
    return df

