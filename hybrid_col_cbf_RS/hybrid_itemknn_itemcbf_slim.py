import numpy as np
import pandas as pd
import gc
from utils.auxUtils import check_matrix, filter_seen, filter_seen_array, buildICMMatrix, buildURMMatrix
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, tracks_data, at=10, k_cbf=10, shrinkage_cbf=10, k_i_i=700, shrinkage_i_i=200,\
                similarity='cosine', tf_idf=True):

        self.k_cbf = k_cbf
        self.k_i_i = k_i_i
        self.at = at
        self.shrinkage_cbf = shrinkage_cbf
        self.shrinkage_i_i = shrinkage_i_i
        self.similarity = similarity
        self.tf_idf = tf_idf
        self.cbf_recommender = CbfRS(tracks_data, self.at, self.k_cbf, self.shrinkage_cbf, tf_idf=self.tf_idf)
        self.col_i_i_recommender = ColBfIIRS(self.at, self.k_i_i, self.shrinkage_i_i, tf_idf=self.tf_idf)

    def fit(self, train_data, lambda_i=0.001, lambda_j=0.001, topK=200, sgd_mode='sgd'):
        print('Fitting...')

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.col_i_i_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)
        self.slim_recommender = SLIM_BPR_Cython(train_data)
        self.slim_recommender.fit(lambda_i=lambda_i, lambda_j=lambda_j, topK=topK, sgd_mode=sgd_mode)

    def recommend(self, playlist_ids, beta=5, gamma=7, delta=10, filter_top_pop=False):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_slim_bpr = self.slim_recommender.get_estimated_ratings()
        '''
        print("CBF")
        print(e_r_cbf[7].data[e_r_cbf[7].data.argsort()[::-1]])
        print("COL_I_I")
        print(e_r_col_i_i[7].data[e_r_col_i_i[7].data.argsort()[::-1]])
        print("COL_U_U")
        print(e_r_col_u_u[7].data[e_r_col_u_u[7].data.argsort()[::-1]])
        print("SLIM")
        print(e_r_slim_bpr[7].data[e_r_slim_bpr[7].data.argsort()[::-1]])
        '''
        estimated_ratings_aux = e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

        # print("Hybrid")
        # print(estimated_ratings_final[7].data[estimated_ratings_final[7].data.argsort()[::-1]])

        estimated_ratings_final = estimated_ratings_aux + e_r_slim_bpr.multiply(delta)

        # print("FINAL")
        # print(estimated_ratings_final[7].data[estimated_ratings_final[7].data.argsort()[::-1]])

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
