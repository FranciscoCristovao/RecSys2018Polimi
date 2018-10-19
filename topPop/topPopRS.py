import numpy as np
import pandas as pd


class TopPopRS:

    top_songs = np.arange(0, 9)
    train_data = pd.DataFrame()

    def __init__(self):
        print("TopPop recommender has been initialized")

    def fit(self, train_data):
        print("Fitting...")
        self.top_songs = train_data['track_id'].value_counts().head(20).index.values
        self.train_data = train_data

    def evaluate(self):
        print("Evaluating..")

    def recommend(self, playlist_ids):

        playlist = {}

        for i in range(10000):
            # the thing we called k in chat
            num_already = 0
            recommended_items = self.top_songs[0:10]
            temp = self.train_data['track_id'].loc[self.train_data['playlist_id'] == playlist_ids['playlist_id'][i]].values

            # todo: improve complexity

            top_songs_mask = np.in1d(self.top_songs, temp, invert=True)

            rec_no_repeat = self.top_songs[top_songs_mask]
            rec_no_repeat = rec_no_repeat[0:10]

            string = ' '.join(str(e) for e in rec_no_repeat)
            playlist.update({playlist_ids['playlist_id'][i]: string})


        print("exit from df")
        dataframe = pd.DataFrame(list(playlist.items()), columns=['playlist_id', 'track_ids'])
        dataframe = dataframe.sort_values(['playlist_id'], ascending=True)

        return dataframe