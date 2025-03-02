import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import filter_seen, buildURMMatrix, normalize_tf_idf, check_matrix
from utils.cosine_similarity import Compute_Similarity_Python
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity as Cosine_Similarity

class ColBfUURS:

    sym = pd.DataFrame()
    urm = pd.DataFrame()
    train_data = pd.DataFrame()

    def __init__(self, at, k=200, shrinkage=50, similarity='cosine', tf_idf=False):

        self.k = k
        # self.cosine = Cosine()
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        self.at = at
        self.tf_idf = tf_idf

    def fit(self, train_data):

        print("Fitting...")

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.urm = buildURMMatrix(train_data)
        if self.tf_idf:
            self.urm = normalize_tf_idf(self.urm.T).T
        self.cosine = Cosine_Similarity(self.urm.T, self.k, self.shrinkage, normalize=True)
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
        return csr_matrix(self.sym.dot(self.urm))

    def get_W(self):
        '''
        W * R = (R.T * W.T).T

        proof:
        a = np.matrix('1 2; 3 4; 5 6')
        b = np.matrix('7 8 9; 10 11 12')
        a * b
        matrix([[27, 30, 33],
                [61, 68, 75],
                [95, 106, 117]])
        (b.T * a.T).T
        matrix([[ 27,  30,  33],
                [ 61,  68,  75],
                [ 95, 106, 117]])
        '''
        return csr_matrix(self.sym.T)

