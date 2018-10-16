import numpy as np
import pandas as pd


class TopPopRS:

    topSongs = np.arange(0, 9)

    def __init__(self):
        print("TopPop recommender has been initialized")

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating..")

    def recommend(self, train_data, playlist_ids):

        topSongs = train_data['track_id'].value_counts().head(20).index.values
        playlist = {}

        for i in range(10000):
            # the thing we called k in chat
            num_already = 0
            recommended_items = topSongs[0:10]
            temp = train_data['track_id'].loc[train_data['playlist_id'] == playlist_ids['playlist_id'][i]].values

            # todo: improve complexity

            topSongs_mask = np.in1d(topSongs, temp, invert=True)

            rec_no_repeat = topSongs[topSongs_mask]
            rec_no_repeat = rec_no_repeat[0:10]
            print(i)
            string = ' '.join(str(e) for e in rec_no_repeat)
            playlist.update({playlist_ids['playlist_id'][i]: string})


        print("exit from df")
        dataframe = pd.DataFrame(list(playlist.items()), columns=['playlist_id', 'track_ids'])
        dataframe = dataframe.sort_values(['playlist_id'], ascending=True)

        return dataframe
