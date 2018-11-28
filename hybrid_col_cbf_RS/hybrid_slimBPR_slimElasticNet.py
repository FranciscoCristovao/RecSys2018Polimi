import numpy as np
import pandas as pd
import gc
from utils.auxUtils import check_matrix, filter_seen, filter_seen_array, buildICMMatrix, buildURMMatrix
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from slimRS.slimElasticNet import SLIMElasticNetRecommender


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, at=10):
        print("Hybrid Slim(s) Initialized")
        self.at = at

    def fit(self, train_data, lambda_i=0.001, lambda_j=0.001, topK_bpr=200, l1_ratio=0.00001,
            topK_elasticNet = 50, sgd_mode='sgd'):
        print('Fitting...')

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.slim_bpr_recommender = SLIM_BPR_Cython(train_data)
        self.slim_bpr_recommender.fit(lambda_i=lambda_i, lambda_j=lambda_j, topK=topK_bpr, sgd_mode=sgd_mode)
        self.slim_elasticNet_recommender = SLIMElasticNetRecommender(train_data)
        self.slim_elasticNet_recommender.fit(l1_ratio=l1_ratio, topK=topK_elasticNet)

    def recommend(self, playlist_ids, filter_top_pop=False):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}
        counter = 0

        e_r_slim_bpr = self.slim_bpr_recommender.get_estimated_ratings()
        e_r_slim_elasticNet = self.slim_elasticNet_recommender.get_estimated_ratings()

        print("SLIM_BPR")
        print(e_r_slim_bpr[7].data[e_r_slim_bpr[7].data.argsort()[::-1]])
        print("SLIM_ElasticNet")
        print(e_r_slim_elasticNet[7].data[e_r_slim_elasticNet[7].data.argsort()[::-1]])

        estimated_ratings_final = e_r_slim_bpr + e_r_slim_elasticNet.multiply(100)

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
