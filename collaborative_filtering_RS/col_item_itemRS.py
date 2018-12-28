import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import filter_seen, buildURMMatrix, normalize_tf_idf, check_matrix
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity
# from utils.cosine_similarity import Compute_Similarity_Python


class ColBfIIRS:

    def __init__(self, at, k=350, shrinkage=400, similarity='cosine', tf_idf=False):

        self.k = k
        # self.cosine = Cosine()
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        self.at = at
        self.tf_idf = tf_idf

    def fit(self, train_data, init_URM=None):

        print("Fitting...")

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        if init_URM is None:
            self.urm = buildURMMatrix(train_data)
        else:
            self.urm = init_URM

        if self.tf_idf:
            self.urm = normalize_tf_idf(self.urm.T).T
        self.cosine = Cosine_Similarity(self.urm, self.k, self.shrinkage, normalize=True)
        # self.cosine = Compute_Similarity_Python(self.urm, self.k, self.shrinkage, normalize=True)
        self.sym = check_matrix(self.cosine.compute_similarity(), 'csr')

    def recommend(self, playlist_ids):

        print("Recommending...")

        final_prediction = {}
        estimated_ratings = csr_matrix(self.urm.dot(self.sym))
        counter = 0

        for k in playlist_ids:
            try:
                row = estimated_ratings[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                top_songs = filter_seen(aux, user_playlist)[:self.at]

                string = ' '.join(str(e) for e in top_songs)
                final_prediction.update({k: string})

                if (counter % 1000) == 0:
                    print("Playlist num", counter, "/10000")
            except IndexError:
                continue
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

    def get_estimated_ratings(self):
        return csr_matrix(self.urm.dot(self.sym))

    def get_sym_matrix(self, weight):
        return check_matrix(self.sym*weight, 'csr')
