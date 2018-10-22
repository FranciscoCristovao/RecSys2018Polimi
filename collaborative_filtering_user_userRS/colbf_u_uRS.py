import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, Cosine
# from utils.cosine_similarity import Cosine_Similarity
# from sklearn.metrics.pairwise import cosine_similarity


class ColBfUURS:

    sym = pd.DataFrame()
    urm = pd.DataFrame()
    helper = Helper()
    train_data = pd.DataFrame()

    def __init__(self):
        print("CBF recommender has been initialized")

    def fit(self, train_data):
        print("Fitting...")
        self.train_data = train_data
        self.urm = self.helper.buildURMMatrix(train_data)
        print("Starting symilarity computation")
        #self.sym = csr_matrix(cosine_similarity(self.urm, dense_output=False))  # self.cos.compute(self.urm, 0)
        # print(self.urm)
        self.sym = Cosine_Similarity(self.urm.T).compute_similarity()
        print(self.sym)
        # print("Symilarity matrix u-u: \n", self.sym)
        print("Sym mat completed")

    def recommend(self, playlist_ids):
        print("Recommending...")
        final_prediction = {}  # pd.DataFrame([])

        print("Transforming into mat_user")
        mat_user = csr_matrix(self.sym).toarray()
        counter = 0
        print("mat_user built")

        for k in playlist_ids:

            row = mat_user[k]
            # print("Playlist with id ", k)
            # print(row)
            # aux contains the indices (track_id) of the most similar songs
            aux = np.argsort(-row)
            # print(aux)
            # top_sym_playlists = aux[:20]

            rec_no_repeat = []
            inc = 0
            no_rep_songs = self.train_data['track_id'].loc[self.train_data['playlist_id'] == k].values
            while len(rec_no_repeat) < 10:
                top_songs = self.train_data['track_id'].loc[self.train_data['playlist_id'] == aux[inc]].values
                songs_mask = np.in1d(top_songs, no_rep_songs, invert=True)
                rec_no_repeat.extend(top_songs[songs_mask][:10])
                inc = inc+1
                # print(k, rec_no_repeat, len(rec_no_repeat))
            rec_no_repeat = rec_no_repeat[:10]
            string = ' '.join(str(e) for e in rec_no_repeat)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def recommend_faster(self, playlist_ids):
        print("Recommending...")
        final_prediction = {}  # pd.DataFrame([])

        print("Transforming into mat_user")
        mat_user = csr_matrix(self.sym)
        counter = 0
        print("mat_user built")

        for k in playlist_ids:

            row = mat_user.getrow(k)

            # aux contains the indices (track_id) of the most similar songs
            sort_indx = np.argsort(-row.data)
            aux = row.indices[sort_indx]

            rec_no_repeat = []
            inc = 0
            no_rep_songs = self.train_data['track_id'].loc[self.train_data['playlist_id'] == k].values
            while len(rec_no_repeat) < 10:
                if inc >= len(aux):
                    print(k, rec_no_repeat)
                    break
                top_songs = self.train_data['track_id'].loc[self.train_data['playlist_id'] == aux[inc]].values
                songs_mask = np.in1d(top_songs, no_rep_songs, invert=True)
                rec_no_repeat.extend(top_songs[songs_mask][:10])
                inc = inc+1

                # print(k, rec_no_repeat, len(rec_no_repeat))
            rec_no_repeat = rec_no_repeat[:10]
            string = ' '.join(str(e) for e in rec_no_repeat)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
