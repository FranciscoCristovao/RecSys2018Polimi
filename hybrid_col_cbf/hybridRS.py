import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, check_matrix, filter_seen
from utils.cosine_similarity_full import Compute_Similarity_Python


class HybridRS:

    helper = Helper()
    train_data = pd.DataFrame()

    def __init__(self, data, at, k=200, shrinkage=0, similarity='cosine'):

        self.k = k
        self.at = at
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        data = data.drop(columns="duration_sec")
        self.icm = self.helper.buildICMMatrix(data)
        print("ICM loaded into the class")

    def fit(self, train_data):
        print('Fitting...')

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.urm = self.helper.buildURMMatrix(train_data)
        self.cosine_cbf = Compute_Similarity_Python(self.icm.T, self.k, self.shrinkage)
        self.cosine_colf = Compute_Similarity_Python(self.urm.T, self.k, self.shrinkage)
        # self.sym = check_matrix(cosine_similarity(self.urm, dense_output=False), 'csr')
        self.sym_items = check_matrix(self.cosine_cbf.compute_similarity(), 'csr')
        self.sym_users = check_matrix(self.cosine_colf.compute_similarity(), 'csr')
        # self.sym = check_matrix(self.cosine.compute(self.urm), 'csr')
        print("Sym mat completed")

    def recommend(self, playlist_ids, alpha):
        print("Recommending...")

        final_prediction = {}
        counter = 0
        # alpha = 0.7  # best until now

        estimated_ratings_cbf = csr_matrix(self.urm.dot(self.sym_items))
        estimated_ratings_colf = csr_matrix(self.sym_users.dot(self.urm))
        estimated_ratings_final = estimated_ratings_colf.multiply(alpha) + estimated_ratings_cbf.multiply(1-alpha)

        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                top_songs = filter_seen(aux, user_playlist)[:self.at]

                string = ' '.join(str(e) for e in top_songs)
                final_prediction.update({k: string})
            except IndexError:
                print("I don't have a value in the test_data")

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
