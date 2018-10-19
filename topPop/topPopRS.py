import numpy as np
import pandas as pd


class TopPopRS:

    top_songs = np.arange(0, 19)
    train_data = pd.DataFrame()

    def __init__(self, train_data):
        print("TopPop recommender has been initialized")
        self.train_data = train_data

    def fit(self):
        print("Fitting...")
        self.top_songs = self.train_data['track_id'].value_counts().head(20).index.values

    def evaluate(self):
        print("Evaluating..")

    def recommend(self, playlist_ids):

        playlist = {}

        for i in playlist_ids:
            # the thing we called k in chat
            num_already = 0
            recommended_items = top_songs[0:10]
            temp = self.train_data['track_id'].loc[self.train_data['playlist_id'] == i].values

            # todo: improve complexity

            top_songs_mask = np.in1d(top_songs, temp, invert=True)

            rec_no_repeat = top_songs[top_songs_mask]
            rec_no_repeat = rec_no_repeat[0:10]
            print(i)
            string = ' '.join(str(e) for e in rec_no_repeat)
            playlist.update({playlist_ids['playlist_id'][i]: string})


        print("exit from df")
        dataframe = pd.DataFrame(list(playlist.items()), columns=['playlist_id', 'track_ids'])
        dataframe = dataframe.sort_values(['playlist_id'], ascending=True)

        return dataframe
