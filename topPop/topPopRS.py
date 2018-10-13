import numpy as np
import pandas as pd
from loader.loader import trainData


class TopPopRS:

    topSongs = np.arange(0, 9)

    def __init__(self):
        print("TopPop recommender has been initialized")

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating..")

    def recommend(self):

        topSongs = trainData['track_id'].value_counts().head(10).index.values

        return topSongs

    def recommend_prop(self, train_data, playlist_ids):

        topSongs = train_data['track_id'].value_counts().head(10).index.values
        string = ' '.join(str(e) for e in topSongs)

        playlist = {}

        for i in range(10000):
            playlist.update({playlist_ids['playlist_id'][i]: string})

        dataframe = pd.DataFrame(playlist.items(), columns=['playlist_id', 'track_ids'])
        dataframe = dataframe.sort_values(['playlist_id'], ascending=True)

        return dataframe
