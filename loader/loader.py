import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Load data
fullData = pd.read_csv('data/train.csv', sep=',')
trainData, testData = train_test_split(fullData, test_size=0.2)
targetData = pd.read_csv('data/target_playlists.csv', sep=',')
tracksData = pd.read_csv('data/tracks.csv', sep=',')

# urm_full_data = pd.read_csv('data/URM.csv', sep=',')


# Store DataFrame in csv
def save_dataframe(path, sep, dataframe):

    dataframe.to_csv(path, index=False, sep=sep)
    print("Successfully built csv..")
