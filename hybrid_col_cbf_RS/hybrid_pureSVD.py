import numpy as np
import pandas as pd
from utils.auxUtils import filter_seen, filter_seen_array, buildURMMatrix
from svdRS.pureSVD import PureSVDRecommender
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from scipy import sparse


class HybridRS:

    train_data = pd.DataFrame()

    def __init__(self, tracks_data, at, k_con=10, k_col_u_u=200, k_col_i_i=700,
                 shrinkage_con=10, shrinkage_col_u_u=50, shrinkage_col_i_i=450, similarity='cosine', tf_idf=True):

        # hybrid parameters
        self.k_con = k_con
        self.k_col_u_u = k_col_u_u
        self.k_col_i_i = k_col_i_i
        self.shrinkage_con = shrinkage_con
        self.shrinkage_col_u_u = shrinkage_col_u_u
        self.shrinkage_col_i_i = shrinkage_col_i_i
        self.tf_idf = tf_idf

        self.at = at
        self.similarity_name = similarity

        self.cbf_recommender = CbfRS(tracks_data, self.at, self.k_con, self.shrinkage_con, tf_idf=self.tf_idf)
        self.col_i_i_recommender = ColBfIIRS(self.at, self.k_col_i_i, self.shrinkage_col_i_i, tf_idf=self.tf_idf)
        self.col_u_u_recommender = ColBfUURS(self.at, self.k_col_u_u, shrinkage_col_u_u, tf_idf=self.tf_idf)

        print("ICM loaded into the class")

    def fit(self, train_data):
        print('Fitting...')

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.col_i_i_recommender.fit(train_data)
        self.col_u_u_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)

        # todo: handle slim init inside init
        self.pureSVD = PureSVDRecommender(train_data)
        self.pureSVD.fit()

    def recommend(self, playlist_ids, alpha=1, beta=5, gamma=7, delta=0.9, filter_top_pop=True):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        row = self.pureSVD.compute_score_SVD(7)

        print("CBF")
        print(e_r_cbf[7].data[e_r_cbf[7].data.argsort()[::-1]])
        print("COL_I_I")
        print(e_r_col_i_i[7].data[e_r_col_i_i[7].data.argsort()[::-1]])
        print("COL_U_U")
        print(e_r_col_u_u[7].data[e_r_col_u_u[7].data.argsort()[::-1]])
        print("pureSVD")
        row.sort()
        print(row[::-1])

        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

        for k in playlist_ids:

            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                mf_row = self.pureSVD.compute_score_SVD(k)
                mf_row = mf_row.argsort()[::-1]
                mf_row_filtered = filter_seen(mf_row, user_playlist)
                # todo: filter top songs
                # mf_row_filtered = filter_seen(mf_row_filtered, self.top_pop_songs)

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)

                top_songs = filter_seen(aux, user_playlist)

                if filter_top_pop:
                    top_songs = list(filter_seen_array(top_songs, self.top_pop_songs)[:int(round(self.at*delta))])
                else:
                    top_songs = list(top_songs[:int(round(self.at*delta))])

                i = 0
                while len(top_songs) < 10 and i < len(mf_row_filtered):
                    el = mf_row_filtered[i]
                    if el not in top_songs:
                        top_songs.append(el)
                    i += 1

                if len(top_songs) < 10:
                    print("Francisco was right once")

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

    def recommend_linear(self, playlist_ids, alpha=1, beta=5, gamma=7, delta=10, filter_top_pop=True):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        # e_r_pureSVD = self.pureSVD.get_estimated_ratings()
        '''
        print("CBF")
        print(e_r_cbf[7].data[e_r_cbf[7].data.argsort()[::-1]])
        print("COL_I_I")
        print(e_r_col_i_i[7].data[e_r_col_i_i[7].data.argsort()[::-1]])
        print("COL_U_U")
        print(e_r_col_u_u[7].data[e_r_col_u_u[7].data.argsort()[::-1]])
        
        print("pureSVD")
        print(e_r_pureSVD[7].data[e_r_pureSVD[7].data.argsort()[::-1]])
        '''
        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)
        print('after sum..')
        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # getting the row from svd
                # try with check matrix..
                mf_row = sparse.csr_matrix(self.pureSVD.compute_score_SVD(k)).multiply(delta)
                # summing it to the row we are considering
                row += mf_row
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

                if len(top_songs) < 10:
                    print("Francisco was right once")

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
