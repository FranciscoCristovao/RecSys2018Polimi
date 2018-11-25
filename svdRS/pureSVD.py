#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 14/06/18
@author: Maurizio Ferrari Dacrema
"""

from utils.auxUtils import check_matrix, buildURMMatrix, filter_seen

from sklearn.decomposition import TruncatedSVD
import scipy.sparse as sps
import pandas as pd
import numpy as np


class PureSVDRecommender():
    """ PureSVDRecommender"""

    RECOMMENDER_NAME = "PureSVDRecommender"

    def __init__(self, train_data, at=10):

        # CSR is faster during evaluation
        self.URM_train = check_matrix(buildURMMatrix(train_data), 'csr')

        self.compute_item_score = self.compute_score_SVD

        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.at = at

    def fit(self, num_factors=470):

        from sklearn.utils.extmath import randomized_svd

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition...")

        self.U, self.Sigma, self.VT = randomized_svd(self.URM_train,
                                                     n_components=num_factors,
                                                     # n_iter=5,
                                                     random_state=None)

        self.s_Vt = sps.diags(self.Sigma) * self.VT

        print(self.RECOMMENDER_NAME + " Computing SVD decomposition... Done!")

        # truncatedSVD = TruncatedSVD(n_components = num_factors)
        #
        # truncatedSVD.fit(self.URM_train)
        #
        # truncatedSVD

        # U, s, Vt =

    def compute_score_SVD(self, user_id_array):

        try:

            item_weights = self.U[user_id_array, :].dot(self.s_Vt)
        except:
            pass
        return item_weights

    def recommend(self, playlist_ids):

        final_prediction = {}
        counter = 0

        for k in playlist_ids:
            try:
                row = self.compute_score_SVD(k)
                # aux contains the indices (track_id) of the most similar songs
                aux = row.argsort()[::-1]
                user_playlist = self.URM_train[k]

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
        return df

    def saveModel(self, folder_path, file_name=None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        data_dict = {
            "U": self.U,
            "Sigma": self.Sigma,
            "VT": self.VT,
            "s_Vt": self.s_Vt
        }

        pickle.dump(data_dict,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete")

    def get_estimated_ratings(self):
        return self.U.dot(self.s_Vt)
