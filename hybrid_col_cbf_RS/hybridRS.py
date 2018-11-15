import numpy as np
import pandas as pd
from utils.auxUtils import filter_seen, filter_seen_array, buildURMMatrix
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS


class HybridRS:

    def __init__(self, tracks_data, at=10, k_cbf=10, shrinkage_cbf=10, k_i_i=700, shrinkage_i_i=200,
                 k_u_u=200, shrinkage_u_u=50, similarity='tanimoto', tf_idf=True):

        self.k_cbf = k_cbf
        self.k_i_i = k_i_i
        self.k_u_u = k_u_u
        self.at = at
        self.shrinkage_cbf = shrinkage_cbf
        self.shrinkage_i_i = shrinkage_i_i
        self.shrinkage_u_u = shrinkage_u_u
        self.similarity = similarity
        self.tf_idf = tf_idf
        self.cbf_recommender = CbfRS(tracks_data, self.at, self.k_cbf, self.shrinkage_cbf, tf_idf=self.tf_idf,
                                     weight_album=1, weight_artist=1)
        self.col_i_i_recommender = ColBfIIRS(self.at, self.k_i_i, self.shrinkage_i_i, tf_idf=self.tf_idf)
        self.col_u_u_recommender = ColBfUURS(self.at, self.k_u_u, self.shrinkage_u_u, tf_idf=self.tf_idf)

    def fit(self, train_data):

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(5).index.values
        self.col_i_i_recommender.fit(train_data)
        self.col_u_u_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)

        print("All systems are fitted")

    def recommend(self, playlist_ids, alpha=1, beta=5, gamma=7, filter_top_pop=False):
        print("Recommending...")

        final_prediction = {}
        counter = 0
        # alpha = 0.7  # best until now

        estimated_ratings_cbf = self.cbf_recommender.get_estimated_ratings()
        estimated_ratings_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        estimated_ratings_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        estimated_ratings_final = estimated_ratings_col_u_u.multiply(alpha)\
                                + estimated_ratings_col_i_i.multiply(beta)\
                                + estimated_ratings_cbf.multiply(gamma)

        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                top_songs = filter_seen(aux, user_playlist)

                if filter_top_pop:
                    top_songs = filter_seen_array(top_songs, self.top_pop_songs)[:self.at]
                else:
                    top_songs = top_songs[:self.at]

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
