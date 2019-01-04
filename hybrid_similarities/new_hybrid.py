import numpy as np
import pandas as pd
import gc
from utils.auxUtils import check_matrix, filter_seen, filter_seen_array, buildICMMatrix, buildURMMatrix
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from svdRS.pureSVD import PureSVDRecommender
from slimRS.slimElasticNet import SLIMElasticNetRecommender
from scipy import sparse
from tqdm import tqdm
from graphBased.p3alpha import P3alphaRecommender
from graphBased.rp3betaRS import RP3betaRecommender


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, tracks_data, at=10, k_cbf=10, shrinkage_cbf=10, k_i_i=700, shrinkage_i_i=200,\
                k_u_u=200, shrinkage_u_u=50, similarity='cosine', tf_idf=True, bm_25=False):

        self.k_cbf = k_cbf
        self.k_i_i = k_i_i
        self.k_u_u = k_u_u
        self.at = at
        self.shrinkage_cbf = shrinkage_cbf
        self.shrinkage_i_i = shrinkage_i_i
        self.shrinkage_u_u = shrinkage_u_u
        self.similarity = similarity
        self.tf_idf = tf_idf
        self.cbf_recommender = CbfRS(tracks_data, self.at, self.k_cbf, self.shrinkage_cbf, tf_idf=self.tf_idf)
        self.col_i_i_recommender = ColBfIIRS(self.at, self.k_i_i, self.shrinkage_i_i, tf_idf=self.tf_idf)
        self.col_u_u_recommender = ColBfUURS(self.at, self.k_u_u, self.shrinkage_u_u, tf_idf=self.tf_idf)


    def fit(self, train_data, lambda_i=0.001, lambda_j=0.001, topK_bpr=200, l1_ratio=0.1,
            topK_elasticNet=300, alpha_elasticNet=0.0002, sgd_mode='sgd'):
        print('Fitting...')
        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.col_i_i_recommender.fit(train_data)
        self.col_u_u_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)

        self.pureSVD = PureSVDRecommender(train_data)
        self.pureSVD.fit()

        self.p3alpha = P3alphaRecommender(train_data)
        self.p3alpha.fit()

        self.rp3beta = RP3betaRecommender(train_data)
        self.rp3beta.fit()

        self.slim_recommender = SLIM_BPR_Cython(train_data)
        self.slim_recommender.fit(lambda_i=lambda_i, lambda_j=lambda_j, topK=topK_bpr, sgd_mode=sgd_mode)

        self.slim_elasticNet_recommender = SLIMElasticNetRecommender(train_data)
        self.slim_elasticNet_recommender.fit(l1_ratio=l1_ratio, topK=topK_elasticNet, alpha=alpha_elasticNet)

    def recommend(self, playlist_ids, alpha=0.2, beta=10, gamma=1, delta=2, omega=30, eta=10, theta=30, sigma=1,
                  filter_top_pop=False):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}

        cbf_sym = self.cbf_recommender.get_sym_matrix(gamma)

        cii_sym = self.col_i_i_recommender.get_sym_matrix(beta)
        p3a_sym = self.p3alpha.get_sym_matrix(theta)
        rp3b_sym = self.rp3beta.get_sym_matrix(sigma)
        slim_sym = self.slim_recommender.get_sym_matrix(delta)
        en_sym = self.slim_elasticNet_recommender.get_sym_matrix(omega)
        sym = cbf_sym + cii_sym + p3a_sym + slim_sym + en_sym + rp3b_sym
        # e_r_ stands for estimated rating
        e_r_hybrid = self.urm*sym
        # print(e_r_hybrid)
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        '''
        e_r_slim_bpr = self.slim_recommender.get_estimated_ratings()
        e_r_slim_elasticNet = self.slim_elasticNet_recommender.get_estimated_ratings()

        '''
        # estimated_ratings_pureSVD = self.pureSVD.U.dot(self.pureSVD.s_Vt)
        # print(estimated_ratings_pureSVD)
        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_hybrid  # + estimated_ratings_pureSVD * eta

        for k in tqdm(playlist_ids):
            try:
                row = estimated_ratings_final[k].toarray()[0] + (self.pureSVD.compute_score_SVD(k)*eta)
                '''
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                '''
                aux = row.argsort()[::-1]

                user_playlist = self.urm[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                top_songs = filter_seen(aux, user_playlist)[:self.at]

                string = ' '.join(str(e) for e in top_songs)
                final_prediction.update({k: string})
            except IndexError:
                print("I don't have a value in the test_data")


        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
