import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from utils.auxUtils import check_matrix, filter_seen, buildICMMatrix, buildURMMatrix
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, data, at, k_con=40, k_col_u_u=200, k_col_i_i=200,
                 shrinkage_con=0, shrinkage_col_u_u=0, shrinkage_col_i_i=0, similarity='jaccard', idf=False):

        # hybrid parameters
        self.k_con = k_con
        self.k_col_u_u = k_col_u_u
        self.k_col_i_i = k_col_i_i
        self.shrinkage_con = shrinkage_con
        self.shrinkage_col_u_u = shrinkage_col_u_u
        self.shrinkage_col_i_i = shrinkage_col_i_i

        self.idf_flag = idf
        self.at = at
        self.similarity_name = similarity
        data = data.drop(columns="duration_sec")
        self.icm = buildICMMatrix(data)
        print("ICM loaded into the class")

    def fit(self, train_data):
        print('Fitting...')

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.urm = buildURMMatrix(train_data)
        # from some tests, looks like k_con optimal = 40 with no particular shrink
        self.cosine_cbf = Cosine_Similarity(self.icm.T, self.k_con, self.shrinkage_con, normalize=True,
                                            mode=self.similarity_name, row_weights=None)
        self.cosine_col_u_u = Cosine_Similarity(self.urm.T, self.k_col_u_u, self.shrinkage_col_u_u, normalize=True,
                                                mode=self.similarity_name, row_weights=None)
        self.cosine_col_i_i = Cosine_Similarity(self.urm, self.k_col_i_i, self.shrinkage_col_i_i, normalize=True,
                                                mode=self.similarity_name, row_weights=None)
        # self.sym = check_matrix(cosine_similarity(self.urm, dense_output=False), 'csr')
        self.sym_cbf = check_matrix(self.cosine_cbf.compute_similarity(), 'csr')
        self.sym_u_u = check_matrix(self.cosine_col_u_u.compute_similarity(), 'csr')
        self.sym_i_i = check_matrix(self.cosine_col_i_i.compute_similarity(), 'csr')
        # self.sym = check_matrix(self.cosine.compute(self.urm), 'csr')
        print("Sym mat completed")

    def recommend(self, playlist_ids, alpha, beta, gamma):
        print("Recommending...")

        final_prediction = {}
        counter = 0
        # alpha = 0.7  # best until now
        # e_r_ stands for estimated rating
        e_r_cbf = csr_matrix(self.urm.dot(self.sym_cbf))
        e_r_col_u_u = csr_matrix(self.sym_u_u.dot(self.urm))
        e_r_col_i_i = csr_matrix(self.urm.dot(self.sym_i_i))
        '''
        print('CbF: ', e_r_cbf.getrow(7))
        print('CuuF: ', e_r_cbf.getrow(7))
        print('CiiF: ', e_r_cbf.getrow(7))
        '''
        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

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
