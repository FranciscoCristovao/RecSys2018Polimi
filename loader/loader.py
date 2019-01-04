import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from utils.auxUtils import split_data_fast
# Load data
full_data = pd.read_csv('data/train.csv', sep=',')
full_data_sequential = pd.read_csv('data/train_sequential.csv', sep=',')
# added random state to have consistent output over songs
# train_data, test_data = train_test_split(full_data, test_size=0.2, random_state=12835)
target_data = pd.read_csv('data/target_playlists.csv', sep=',')
tracks_data = pd.read_csv('data/tracks.csv', sep=',')
# train_data, test_data = train_test_split(full_data, test_size=0.2)
# train_data, test_data = split_data_fast(full_data, full_data_sequential, target_data, test_size=0.2)

train_data = pd.read_csv('data/train_4.csv', sep=',')
test_data = pd.read_csv('data/test_4.csv', sep=',')

# urm_full_data = pd.read_csv('data/URM.csv', sep=',')


# Store DataFrame in csv
def save_dataframe(path, sep=',', dataframe=None):
    dataframe.to_csv(path, index=False, sep=sep)
    print("Successfully built csv..")
