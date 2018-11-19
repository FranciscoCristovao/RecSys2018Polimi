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

    def fit(self, train_data, sgd_mode='sgd'):
        print('Fitting...')

        self.urm = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.col_i_i_recommender.fit(train_data)
        self.col_u_u_recommender.fit(train_data)
        self.cbf_recommender.fit(train_data)

        # todo: handle slim init inside init
        self.slim_rs = SLIM_BPR_Cython(train_data)
        self.slim_rs.fit(sgd_mode=sgd_mode)

    def recommend(self, playlist_ids, alpha=1, beta=5, gamma=7, delta=0.9, filter_top_pop=True):
        print("Recommending... Am I filtering top_top songs?", filter_top_pop)

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        e_r_slim_bpr = self.slim_rs.get_estimated_ratings()
        '''
        print(e_r_cbf[7].data[e_r_cbf[7].data.argsort()[::-1]])
        print(e_r_col_i_i[7].data[e_r_col_i_i[7].data.argsort()[::-1]])
        print(e_r_col_u_u[7].data[e_r_col_u_u[7].data.argsort()[::-1]])
        '''
        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                slim_bpr_row = e_r_slim_bpr[k]
                slim_bpr_index = slim_bpr_row.data.argsort()[::-1]
                slim_bpr_aux = slim_bpr_row.indices[slim_bpr_index]
                slim_bpr_row_filtered = filter_seen(slim_bpr_aux, user_playlist)

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)

                top_songs = filter_seen(aux, user_playlist)

                if filter_top_pop:
                    top_songs = list(filter_seen_array(top_songs, self.top_pop_songs)[:int(round(self.at*delta))])
                else:
                    top_songs = list(top_songs[:int(round(self.at*delta))])

                i = 0
                while len(top_songs) < 10 and i < len(slim_bpr_row_filtered):
                    el = slim_bpr_row_filtered[i]
                    if el not in top_songs:
                        top_songs.append(el)
                    i += 1

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

    def recommend_probability(self, playlist_ids, alpha, beta, gamma, p_treshold=0.9):
        print("Recommending...")

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()
        e_r_slim_bpr = self.slim_rs.get_estimated_ratings()

        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta) + e_r_cbf.multiply(gamma)

        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                slim_bpr_row = e_r_slim_bpr[k]
                slim_bpr_index = slim_bpr_row.data.argsort()[::-1]
                slim_bpr_aux = slim_bpr_row.indices[slim_bpr_index]
                slim_bpr_row_filtered = filter_seen(slim_bpr_aux, user_playlist)

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)

                temp_top_songs = list(filter_seen(aux, user_playlist)[:self.at])

                i = 0
                m = 0
                top_songs = []
                while len(top_songs) < self.at:
                    # todo: we could try using more sophisticated distribution
                    p = np.random.uniform(low=0.0, high=1.0)
                    if p < p_treshold:
                        if i < len(temp_top_songs) and temp_top_songs[i] not in top_songs:
                            top_songs.append(temp_top_songs[i])
                        i += 1
                    else:
                        # draw first non seen element
                        while m < len(slim_bpr_row_filtered) and slim_bpr_row_filtered[m] in top_songs:
                            m += 1
                        if m < len(slim_bpr_row_filtered):
                            top_songs.append(slim_bpr_row_filtered[m])

                # no need to filter playlist songs ( already did) nor to take top k songs ( while assures 10 )

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

    def recommend_reverse(self, playlist_ids, alpha=1, beta=5, gamma=7, delta=0.9):
        print("Recommending...")
        gc.collect()

        final_prediction = {}
        counter = 0

        # e_r_ stands for estimated rating
        e_r_col_i_i = self.col_i_i_recommender.get_estimated_ratings()
        e_r_col_u_u = self.col_u_u_recommender.get_estimated_ratings()

        estimated_ratings_final = e_r_col_u_u.multiply(alpha) + e_r_col_i_i.multiply(beta)
        e_r_col_i_i = []
        e_r_col_u_u = []
        gc.collect()
        e_r_cbf = self.cbf_recommender.get_estimated_ratings()
        estimated_ratings_final += e_r_cbf.multiply(gamma)

        e_r_cbf = []
        gc.collect()
        e_r_slim_bpr = self.slim_rs.get_estimated_ratings()
        '''
        print(e_r_cbf[7].data[e_r_cbf[7].data.argsort()[::-1]])
        print(e_r_col_i_i[7].data[e_r_col_i_i[7].data.argsort()[::-1]])
        print(e_r_col_u_u[7].data[e_r_col_u_u[7].data.argsort()[::-1]])
        '''


        for k in playlist_ids:
            try:
                row = estimated_ratings_final[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.urm[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                aux_filtered = list(filter_seen(aux, user_playlist))

                slim_bpr_row = e_r_slim_bpr[k]
                slim_bpr_index = slim_bpr_row.data.argsort()[::-1]
                slim_bpr_aux = slim_bpr_row.indices[slim_bpr_index]
                slim_bpr_row_filtered = filter_seen(slim_bpr_aux, user_playlist)

                top_songs = list(slim_bpr_row_filtered[:int(round(self.at * (1 - delta)))])

                i = 0
                while len(top_songs) < 10 and i < len(aux_filtered):
                    el = aux_filtered[i]
                    if el not in top_songs:
                        top_songs.append(el)
                    i += 1

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

