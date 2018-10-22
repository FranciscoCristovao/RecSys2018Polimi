import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, Cosine
from utils.cosine_similarity import Cosine_Similarity

class CbfRS:

    icm = pd.DataFrame()
    sym = pd.DataFrame()
    urm = pd.DataFrame()
    helper = Helper()
    train_data = pd.DataFrame()

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
        # Here we have 3 different ways to compute the similarities
        # self.sym = csr_matrix(self.icm.dot(self.icm.T))
        self.sym = Cosine_Similarity(self.icm.T).compute_similarity()
        # self.sym = self.cos.compute(self.icm, 0)
        # self.sym = csr_matrix(cosine_similarity(self.icm, self.icm))
        # print(self.sym)
        print("Sym correctly loaded")

        self.urm = self.helper.buildURMMatrix(train_data)

    def recommend(self, playlist_ids):
        print("Recommending...")

        final_prediction = {}  # pd.DataFrame([])

        print("STARTING ESTIMATION")
        # add ravel() ?
        estimated_ratings = csr_matrix(self.urm.dot(self.sym)).toarray()
        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]

            # aux contains the indices (track_id) of the most similar songs
            aux = np.argsort(-row)
            top_songs = aux[:50]

            temp = self.train_data['track_id'].loc[self.train_data['playlist_id'] == k].values
            top_songs_mask = np.in1d(top_songs, temp, invert=True)
            rec_no_repeat = top_songs[top_songs_mask][:10]
            if len(rec_no_repeat) < 10:
                print("playlist k not enough long", k)
            string = ' '.join(str(e) for e in rec_no_repeat)
            final_prediction.update({k: string})

            if(counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
