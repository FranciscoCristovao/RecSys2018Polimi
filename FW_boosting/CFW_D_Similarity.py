#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/17
@author: Maurizio Ferrari Dacrema
"""
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity
from utils.auxUtils import check_matrix, filter_seen
from scipy.sparse import csr_matrix


from scipy.sparse import linalg
import time, sys
import numpy as np
import pandas as pd


class CFW_D_Similarity_Linalg():
    RECOMMENDER_NAME = "CFW_D_Similarity_Linalg"

    def __init__(self, URM_train, ICM, S_matrix_target, train_data):

        super(CFW_D_Similarity_Linalg, self).__init__()


        if (URM_train.shape[1] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. URM contains {} but ICM contains {}".format(URM_train.shape[1],
                                                                                                          ICM.shape[0]))

        if(S_matrix_target.shape[0] != S_matrix_target.shape[1]):
            raise ValueError("Items imilarity matrix is not square: rows are {}, columns are {}".format(S_matrix_target.shape[0],
                                                                                                        S_matrix_target.shape[1]))

        if(S_matrix_target.shape[0] != ICM.shape[0]):
            raise ValueError("Number of items not consistent. S_matrix contains {} but ICM contains {}".format(S_matrix_target.shape[0],
                                                                                                          ICM.shape[0]))

        self.URM_train = check_matrix(URM_train, 'csr')
        self.S_matrix_target = check_matrix(S_matrix_target, 'csr')
        self.ICM = check_matrix(ICM, 'csr')
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values
        self.n_items = self.URM_train.shape[1]
        self.n_users = self.URM_train.shape[0]
        self.n_features = self.ICM.shape[1]

        self.sparse_weights = True

    def _writeLog(self, string):

        print(string)
        sys.stdout.flush()
        sys.stderr.flush()

        if self.logFile is not None:
            self.logFile.write(string + "\n")
            self.logFile.flush()

    def _generateTrainData_low_ram(self):

        print(self.RECOMMENDER_NAME + ": Generating train data")

        start_time_batch = time.time()


        # Here is important only the structure
        self.similarity = Cosine_Similarity(self.ICM.T, shrink=0, topK=self.topK, normalize=False)
        S_matrix_contentKNN = self.similarity.compute_similarity()
        S_matrix_contentKNN = check_matrix(S_matrix_contentKNN, "csr")


        self._writeLog(self.RECOMMENDER_NAME + ": Collaborative S density: {:.2E}, nonzero cells {}".format(
            self.S_matrix_target.nnz/self.S_matrix_target.shape[0]**2, self.S_matrix_target.nnz))

        self._writeLog(self.RECOMMENDER_NAME + ": Content S density: {:.2E}, nonzero cells {}".format(
            S_matrix_contentKNN.nnz/S_matrix_contentKNN.shape[0]**2, S_matrix_contentKNN.nnz))


        if self.normalize_similarity:

            # Compute sum of squared
            sum_of_squared_features = np.array(self.ICM.T.power(2).sum(axis=0)).ravel()
            sum_of_squared_features = np.sqrt(sum_of_squared_features)



        num_common_coordinates = 0

        estimated_n_samples = int(S_matrix_contentKNN.nnz*(1+self.add_zeros_quota)*1.2)

        self.row_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.col_list = np.zeros(estimated_n_samples, dtype=np.int32)
        self.data_list = np.zeros(estimated_n_samples, dtype=np.float64)

        num_samples = 0

        for row_index in range(self.n_items):

            start_pos_content = S_matrix_contentKNN.indptr[row_index]
            end_pos_content = S_matrix_contentKNN.indptr[row_index+1]

            content_coordinates = S_matrix_contentKNN.indices[start_pos_content:end_pos_content]

            start_pos_target = self.S_matrix_target.indptr[row_index]
            end_pos_target = self.S_matrix_target.indptr[row_index+1]

            target_coordinates = self.S_matrix_target.indices[start_pos_target:end_pos_target]

            # Chech whether the content coordinate is associated to a non zero target value
            # If true, the content coordinate has a collaborative non-zero value
            # if false, the content coordinate has a collaborative zero value
            is_common = np.in1d(content_coordinates, target_coordinates)

            num_common_in_current_row = is_common.sum()
            num_common_coordinates += num_common_in_current_row

            for index in range(len(is_common)):

                if num_samples == estimated_n_samples:
                    dataBlock = 1000000
                    self.row_list = np.concatenate((self.row_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.col_list = np.concatenate((self.col_list, np.zeros(dataBlock, dtype=np.int32)))
                    self.data_list = np.concatenate((self.data_list, np.zeros(dataBlock, dtype=np.float64)))

                if is_common[index]:
                    # If cell exists in target matrix, add its value
                    # Otherwise it will remain zero with a certain probability

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index

                    new_data_value = self.S_matrix_target[row_index, col_index]

                    if self.normalize_similarity:
                        new_data_value *= sum_of_squared_features[row_index]*sum_of_squared_features[col_index]

                    self.data_list[num_samples] = new_data_value

                    num_samples += 1

                elif np.random.rand() <= self.add_zeros_quota:

                    col_index = content_coordinates[index]

                    self.row_list[num_samples] = row_index
                    self.col_list[num_samples] = col_index
                    self.data_list[num_samples] = 0.0

                    num_samples += 1

            if time.time() - start_time_batch > 30 or num_samples == S_matrix_contentKNN.nnz*(1+self.add_zeros_quota):

                print(self.RECOMMENDER_NAME + ": Generating train data. Sample {} ( {:.2f} %) ".format(
                    num_samples, num_samples/ S_matrix_contentKNN.nnz*(1+self.add_zeros_quota) *100))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()

        self._writeLog(self.RECOMMENDER_NAME + ": Content S structure has {} out of {} ( {:.2f}%) nonzero collaborative cells".format(
            num_common_coordinates, S_matrix_contentKNN.nnz, num_common_coordinates/S_matrix_contentKNN.nnz*100))

        # Discard extra cells at the left of the array
        self.row_list = self.row_list[:num_samples]
        self.col_list = self.col_list[:num_samples]
        self.data_list = self.data_list[:num_samples]

        data_nnz = sum(np.array(self.data_list) != 0)
        data_sum = sum(self.data_list)

        collaborative_nnz = self.S_matrix_target.nnz
        collaborative_sum = sum(self.S_matrix_target.data)

        self._writeLog(self.RECOMMENDER_NAME + ": Nonzero collaborative cell sum is: {:.2E}, average is: {:.2E}, "
                      "average over all collaborative data is {:.2E}".format(
                      data_sum, data_sum/data_nnz, collaborative_sum/collaborative_nnz))

    def fit(self, show_max_performance = False, logFile = None, loss_tolerance = 1e-6,
            iteration_limit = 50000, damp_coeff=0.0, topK = 300, add_zeros_quota = 0.0, normalize_similarity = False):

        self.logFile = logFile
        self.normalize_similarity = normalize_similarity

        self.add_zeros_quota = add_zeros_quota
        self.topK = topK

        self._generateTrainData_low_ram()


        commonFeatures = self.ICM[self.row_list].multiply(self.ICM[self.col_list])

        linalg_result = linalg.lsqr(commonFeatures, self.data_list, show = False, atol=loss_tolerance, btol=loss_tolerance,
                          iter_lim = iteration_limit, damp=damp_coeff)

        # res = linalg.lsmr(commonFeatures, self.data_list, show = False, atol=loss_tolerance, btol=loss_tolerance,
        #                   maxiter = iteration_limit, damp=damp_coeff)

        self.D_incremental = linalg_result[0].copy()
        self.D_best = linalg_result[0].copy()
        self.epochs_best = 0
        self.loss = linalg_result[3]
        self._compute_W_sparse()

    def _compute_W_sparse(self, use_incremental = False):

        if use_incremental:
            feature_weights = self.D_incremental
        else:
            feature_weights = self.D_best

        self.similarity = Cosine_Similarity(self.ICM.T, shrink=0, topK=self.topK,
                                            normalize=self.normalize_similarity, row_weights=feature_weights)

        self.W_sparse = self.similarity.compute_similarity()
        self.sparse_weights = True

    def saveModel(self, folder_path, file_name = None):

        import pickle

        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        print("{}: Saving model in file '{}'".format(self.RECOMMENDER_NAME, folder_path + file_name))

        dictionary_to_save = {
            "D_best":self.D_best,
            "topK":self.topK,
            "sparse_weights":self.sparse_weights,
            "W_sparse":self.W_sparse,
            "normalize_similarity":self.normalize_similarity
        }

        pickle.dump(dictionary_to_save,
                    open(folder_path + file_name, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        print("{}: Saving complete".format(self.RECOMMENDER_NAME))

    def recommend(self, playlist_ids):
        print("Recommending...")

        final_prediction = {}
        estimated_ratings = csr_matrix(self.URM_train.dot(self.W_sparse))
        counter = 0

        for k in playlist_ids:
            try:
                row = estimated_ratings[k]
                # aux contains the indices (track_id) of the most similar songs
                indx = row.data.argsort()[::-1]
                aux = row.indices[indx]
                user_playlist = self.URM_train[k]

                aux = np.concatenate((aux, self.top_pop_songs), axis=None)
                top_songs = filter_seen(aux, user_playlist)[:10]

                string = ' '.join(str(e) for e in top_songs)
                final_prediction.update({k: string})

                if (counter % 1000) == 0:
                    print("Playlist num", counter, "/10000")
            except IndexError:
                continue
            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df
