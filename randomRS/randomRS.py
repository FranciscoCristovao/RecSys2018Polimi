import random
import pandas as pd

class RandomRS:
    def __init__(self):
        print("Random rec has been initialized")

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating..")

    def recommend_old(self):
        playlist = []

        for i in range(10000):
            temp = []
            for j in range(10):
                temp.append(random.randint(1, 20600))
            playlist.append(temp)
        # 20600 is the number of songs
        return playlist

    def recommend(self, playlist_ids):
        playlist = {}
        pls = []
        for i in range(10000):
            temp = []
            pls.append(playlist_ids['playlist_id'][i])
            for j in range(10):
                temp.append(random.randint(1, 20600))
            string = ' '.join(str(e) for e in temp)
            playlist.update({playlist_ids['playlist_id'][i]: string})
        #print(playlist)
        dataframe = pd.DataFrame(playlist.items(), columns=['playlist_id', 'track_ids'])

        print(dataframe.head(10))
        # 20600 is the number of songs
        return dataframe