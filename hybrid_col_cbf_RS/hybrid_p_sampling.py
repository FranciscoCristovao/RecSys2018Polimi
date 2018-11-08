import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from utils.auxUtils import check_matrix, filter_seen, buildICMMatrix, buildURMMatrix, filter_seen_array
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, data, at, k_con=40, k_col_u_u=200, k_col_i_i=200,
                 shrinkage_con=0, shrinkage_col_u_u=0, shrinkage_col_i_i=0, similarity='cosine'):

        # hybrid parameters
        self.k_con = k_con
        self.k_col_u_u = k_col_u_u
        self.k_col_i_i = k_col_i_i
        self.shrinkage_con = shrinkage_con
        self.shrinkage_col_u_u = shrinkage_col_u_u
        self.shrinkage_col_i_i = shrinkage_col_i_i

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

    def recommend(self, playlist_ids, t, alpha, beta, gamma):
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
        # todo: sample alternatively from u_u, i_i
        # estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

        for k in playlist_ids:
            try:
                row_cbf = e_r_cbf[k]
                row_col_u_u = e_r_col_u_u[k]
                row_col_i_i = e_r_col_i_i[k]
                user_playlist = self.urm[k]

                # u_u
                indx_u_u = row_col_u_u.data.argsort()[::-1]
                aux_col_u_u = row_col_u_u.indices[indx_u_u]

                # i_i
                indx_i_i = row_col_i_i.data.argsort()[::-1]
                aux_col_i_i = row_col_i_i.indices[indx_i_i]

                # content
                indx_cbf = row_cbf.data.argsort()[::-1]
                aux_cbf = row_cbf.indices[indx_cbf]

                i = 0
                j = 0
                m = 0
                count = 0
                top_songs = []
                t1 = t
                # t2 = 6/11 + t1
                while (i < len(aux_col_u_u) or j < len(aux_col_i_i) or m < len(aux_cbf)) and count < 100:
                    #p = np.randomimp.uniform(low=0.0, high=1.0)
                    p = np.random.normal(loc=0.5, scale=0.5, size=None)
                    if 0 < p < 0.3 or t1 < p < 1:
                        if i < len(aux_col_u_u) and aux_col_u_u[i] not in top_songs:
                            top_songs.append(aux_col_u_u[i])
                        i += 1
                    elif 0 < p < t1:
                        if j < len(aux_col_i_i) and aux_col_i_i[j] not in top_songs:
                            top_songs.append(aux_col_i_i[j])
                        j += 1
                    else:
                        if m < len(aux_cbf) and aux_cbf[m] not in top_songs:
                            top_songs.append(aux_cbf[m])
                        m += 1
                    count += 1

                recommended_songs = filter_seen_array(np.array(top_songs), user_playlist.data)[:self.at]

                if len(recommended_songs) < 10:
                    print("Francisco was right once")

                string = ' '.join(str(e) for e in recommended_songs)
                final_prediction.update({k: string})
            except IndexError:
                print("I don't have a value in the test_data")

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
