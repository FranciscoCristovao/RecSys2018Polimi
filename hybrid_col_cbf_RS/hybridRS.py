import numpy as np
import pandas as pd
from utils.auxUtils import filter_seen, buildURMMatrix
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS


class HybridRS:

    def __init__(self, tracks_data, at, k_cbf=40, k_collab=200, shrinkage=0, similarity='cosine', tf_idf=False):

        self.k_cbf = k_cbf
        self.k_collab = k_collab
        self.at = at
        self.shrinkage = shrinkage
        self.similarity = similarity
        self.tf_idf = tf_idf
        self.cbf_recommender = CbfRS(tracks_data, self.at, self.k_cbf, self.shrinkage, tf_idf=self.tf_idf)
        self.collab_recommender = ColBfIIRS(self.at, self.k_collab, self.shrinkage, tf_idf=self.tf_idf)

    def fit(self, train_data):

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.collab_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)

        print("All systems are fitted")

    def recommend(self, playlist_ids, alpha):
        print("Recommending...")

        final_prediction = {}
        counter = 0
        # alpha = 0.7  # best until now

        estimated_ratings_cbf = self.cbf_recommender.get_estimated_ratings()
        estimated_ratings_colf = self.collab_recommender.get_estimated_ratings()
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
