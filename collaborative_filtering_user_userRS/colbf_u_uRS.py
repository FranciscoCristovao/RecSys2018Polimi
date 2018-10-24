import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, filter_seen
from utils.cosine_similarity_full import Compute_Similarity_Python, check_matrix
from sklearn.metrics.pairwise import cosine_similarity
from utils.cosine_similarity import Cosine


class ColBfUURS:

    sym = pd.DataFrame()
    urm = pd.DataFrame()
    helper = Helper()
    train_data = pd.DataFrame()

    def __init__(self, at, k=200, shrinkage=0, similarity='cosine'):

        self.k = k
        # self.cosine = Cosine()
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        self.at = at

    def fit(self, train_data):
        print("Fitting...")

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.urm = self.helper.buildURMMatrix(train_data)
        self.cosine = Compute_Similarity_Python(self.urm.T, self.k, self.shrinkage)
        # self.sym = check_matrix(cosine_similarity(self.urm, dense_output=False), 'csr')
        self.sym = check_matrix(self.cosine.compute_similarity(), 'csr')
        # self.sym = check_matrix(self.cosine.compute(self.urm), 'csr')
        print("Sym mat completed")

    def recommend(self, playlist_ids):
        print("Recommending...")
        final_prediction = {}

        estimated_ratings = csr_matrix(self.sym.dot(self.urm))
        counter = 0

        for k in playlist_ids:

            row = estimated_ratings.getrow(k)
            # aux contains the indices (track_id) of the most similar songs
            indx = row.data.argsort()[::-1]
            aux = row.indices[indx]
            user_playlist = self.urm[k]

            top_songs = filter_seen(user_playlist, aux)[:self.at]

            if len(top_songs) < self.at:
                # todo: check this
                top_songs = np.concatenate((top_songs, self.top_pop_songs), axis=None)[:self.at]

                print("Francisco was right once at least")

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def recommend_single(self, k):
        print("Playlist num: ", k, "/50440")
        row = self.sym.getrow(k)
        # compute prediction
        est_row = csr_matrix(row.dot(self.urm))
        # retrieve the index
        # print(est_row)
        indx = est_row.data.argsort()[::-1]
        aux = est_row.indices[indx]

        user_playlist = self.urm[k]
        # filter the songs
        top_songs = filter_seen(user_playlist, aux)[:self.at]

        return top_songs

    def recommend_slower(self, playlist_ids):
        print("Recommending...")
        final_prediction = {}

        estimated_ratings = csr_matrix(self.sym.dot(self.urm)).toarray()
        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]

            # aux contains the indices (track_id) of the most similar songs
            aux = row.argsort()[::-1]
            user_playlist = self.urm[k]

            top_songs = filter_seen(user_playlist, aux)[:self.at]

            if len(top_songs) < self.at:
                print("Francisco was right once at least")

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    '''
        def recommend_slower(self, playlist_ids):
        print("Recommending...")

        final_prediction = {}

        estimated_ratings = csr_matrix(self.sym.dot(self.urm)).toarray()
        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]
            # aux contains the indices (track_id) of the most similar songs
            aux = row.argsort()[::-1]
            user_playlist = self.urm[k]

            top_songs = filter_seen(user_playlist, aux)[:10]

            if len(top_songs) < 10:
                print("Francisco was right once at least")

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
    '''
