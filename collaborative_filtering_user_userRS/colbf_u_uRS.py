import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, check_matrix, filter_seen
from utils.cosine_similarity import Cosine

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
        # self.sym = csr_matrix(cosine_similarity(self.urm, dense_output=False))  # self.cos.compute(self.urm, 0)
        # print(self.urm)
        self.sym = check_matrix(Cosine().compute(self.urm), 'csr')
        print(type(self.sym))
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
        mat_user = self.sym
        counter = 0
        print("mat_user built")

        for k in playlist_ids:

            row = mat_user.getrow(k)

            # aux contains the indices (track_id) of the most similar songs
            sort_indx = row.data.argsort()[::-1]
            user_similar_songs = row.indices[sort_indx]

            user_playlist = self.urm[k]
            top_songs = filter_seen(user_playlist, user_similar_songs)[:10]

            if len(top_songs) < 10:
                print("Francisco was right once at least")
                top_songs = [8956, 10848, 5606, 15578, 10496, 17239, 13980, 2674, 18266, 2272]

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
