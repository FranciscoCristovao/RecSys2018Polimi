import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper

class cbfRS:

    icm = pd.DataFrame()
    sym = pd.DataFrame()
    urm = pd.DataFrame()
    helper = Helper()

    def __init__(self, data):
        print("CBF recommender has been initialized")

        data = data.drop(columns="duration_sec")
        self.icm = self.helper.buildICMMatrix(data)
        # print(self.icm.todense())
        print("ICM loaded into the class")


    def fit(self, train_data):
        print("Fitting...")
        self.train_data = train_data
        # it was a numpy array, i transformed it into a csr matrix
        self.sym = csr_matrix(cosine_similarity(self.icm, self.icm))
        print("Sym correctly loaded")

        self.urm = self.helper.buildURMMatrix(train_data)


    def recommend(self, playlist_ids):
        print("Recommending...")

        pred = {}  # pd.DataFrame([])

        print("STARTING ESTIMATION")
        estimated_ratings = csr_matrix(self.urm.dot(self.sym))
        counter = 0

        for k in playlist_ids:

            '''print("Ratings for the first row, reversed:",
                  estimated_ratings.data[estimated_ratings.indptr[k]:estimated_ratings.indptr[k+1]])'''
            row = estimated_ratings[k]
            # print("Index", k)
            # print("Row.data", row.data)
            # aux contains the indices (track_id) of the most similar songs
            aux = np.argsort(-row.data)
            # print("Row.data sorted", aux)
            top_songs = row.indices[aux[:20]]
            # print("Top Songs ", top_songs)
            # print("Music to suggest ", top_songs)

            temp = self.train_data['track_id'].loc[self.train_data['playlist_id'] == k].values
            # print("Songs in the playlist", temp)

            top_songs_mask = np.in1d(top_songs, temp, invert=True)

            rec_no_repeat = top_songs[top_songs_mask]
            rec_no_repeat = rec_no_repeat[0:10]

            string = ' '.join(str(e) for e in rec_no_repeat)
            pred.update({k: string})
            print("Playlist num", counter, "/10000")
            counter += 1

        df = pd.DataFrame(list(pred.items()), columns = ['playlist_id', 'track_ids'])
        print(df)
        return df
