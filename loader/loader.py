import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
full_data = pd.read_csv('data/train.csv', sep=',')
# added random state to have consistent output over songs
train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=12345)
# train_data, test_data = train_test_split(full_data, test_size=0.2)
target_data = pd.read_csv('data/target_playlists.csv', sep=',')
tracks_data = pd.read_csv('data/tracks.csv', sep=',')

# urm_full_data = pd.read_csv('data/URM.csv', sep=',')


# Store DataFrame in csv
def save_dataframe(path, sep, dataframe):

    dataframe.to_csv(path, index=False, sep=sep)
    print("Successfully built csv..")
