#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 28 June 2017

@author: Maurizio Ferrari Dacrema
"""

import sys
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.sparse import csr_matrix
from utils.auxUtils import check_matrix, filter_seen, buildURMMatrix
# from Base.Recommender import Recommender


class SLIM_BPR():
    """
    This class is a python porting of the BPRSLIM algorithm in MyMediaLite written in C#
    The code is identical with no optimizations
    """

    def __init__(self, train_data, lambda_i=0.0025, lambda_j=0.00025, learning_rate=0.05):
        # super(SLIM_BPR, self).__init__()
        # was his URM csr?
        self.URM_train = check_matrix(buildURMMatrix(train_data), 'csr')
        self.n_playlist = self.URM_train.shape[0]
        self.n_songs = self.URM_train.shape[1]
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate

        self.normalize = False
        self.sparse_weights = False

    def updateFactors(self, playlist_id, pos_item_id, neg_item_id):

        # Calculate current predicted score
        playlistSeenItems = self.URM_train[playlist_id].indices
        prediction = 0

        for playlistSeenItem in playlistSeenItems:
            prediction += self.S[pos_item_id, playlistSeenItem] - self.S[neg_item_id, playlistSeenItem]

        x_uij = prediction
        logisticFunction = expit(-x_uij)

        # Update similarities for all items except those sampled
        for playlistSeenItem in playlistSeenItems:

            # For positive item is PLUS logistic minus lambda*S
            if pos_item_id != playlistSeenItem:
                update = logisticFunction - self.lambda_i*self.S[pos_item_id, playlistSeenItem]
                self.S[pos_item_id, playlistSeenItem] += self.learning_rate*update

            # For positive item is MINUS logistic minus lambda*S
            if neg_item_id != playlistSeenItem:
                update = - logisticFunction - self.lambda_j*self.S[neg_item_id, playlistSeenItem]
                self.S[neg_item_id, playlistSeenItem] += self.learning_rate*update

    def fit(self, epochs=30):
        """
        Train SLIM wit BPR. If the model was already trained, overwrites matrix S
        :param epochs:
        :return: -
        """

        # Initialize similarity with random values and zero-out diagonal
        self.S = np.random.random((self.n_songs, self.n_songs)).astype('float32')
        self.S[np.arange(self.n_songs), np.arange(self.n_songs)] = 0

        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            self.epochIteration()
            print("Epoch {} of {} complete in {:.2f} minutes"
                  .format(currentEpoch+1, epochs, float(time.time()-start_time_epoch)/60))

        print("Train completed in {:.2f} minutes".format(float(time.time()-start_time_train)/60))

        # The similarity matrix is learnt row-wise
        # To be used in the product URM*S must be transposed to be column-wise
        self.W = self.S.T
        # self.W = csr_matrix(self.W)

        del self.S

    def recommend(self, playlist_ids):

        print("Recommending...")

        final_prediction = {}
        # what dimension does W have?

        self.W = csr_matrix(self.W, shape=(self.n_songs, self.n_songs))
        estimated_ratings = check_matrix(self.URM_train.dot(self.W), 'csr')

        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]
            # aux contains the indices (track_id) of the most similar songs
            indx = row.data.argsort()[::-1]
            aux = row.indices[indx]
            user_playlist = self.URM_train[k]

            # aux = np.concatenate((aux, self.top_pop_songs), axis=None)
            top_songs = filter_seen(aux, user_playlist)[:10]

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def epochIteration(self):

        # Get number of available interactions
        numPositiveIteractions = self.URM_train.nnz

        start_time = time.time()

        # Uniform user sampling without replacement
        for numSample in range(numPositiveIteractions):

            user_id, pos_item_id, neg_item_id = self.sampleTriple()
            self.updateFactors(user_id, pos_item_id, neg_item_id)

            if(numSample % 10000 == 0):
                print("Processed {} ( {:.2f}% ) in {:.4f} seconds".format(numSample,
                                  100.0* float(numSample)/numPositiveIteractions,
                                  time.time()-start_time))

                sys.stderr.flush()

                start_time = time.time()

    def sampleUser(self):
        """
        Sample a user that has viewed at least one and not all items
        :return: user_id
        """
        while True:

            user_id = np.random.randint(0, self.n_playlist)
            numSeenItems = self.URM_train[user_id].nnz

            if(numSeenItems >0 and numSeenItems<self.n_songs):
                return user_id



    def sampleItemPair(self, user_id):
        """
        Returns for the given user a random seen item and a random not seen item
        :param user_id:
        :return: pos_item_id, neg_item_id
        """

        userSeenItems = self.URM_train[user_id].indices

        pos_item_id = userSeenItems[np.random.randint(0, len(userSeenItems))]

        while(True):

            neg_item_id = np.random.randint(0, self.n_songs)

            if neg_item_id not in userSeenItems:
                return pos_item_id, neg_item_id


    def sampleTriple(self):
        """
        Randomly samples a user and then samples randomly a seen and not seen item
        :return: user_id, pos_item_id, neg_item_id
        """

        user_id = self.sampleUser()
        pos_item_id, neg_item_id = self.sampleItemPair(user_id)

        return user_id, pos_item_id, neg_item_id

'''
    def recommend_single(self, playlist_id, at=None, exclude_seen=True):

        # compute the scores using the dot product
        songs_playlist = self.URM_train[playlist_id]
        scores = songs_playlist.dot(self.W)
        scores = scores.toarray()

        # rank items
        ranking = scores.argsort()[::-1].squeeze()
        if exclude_seen:
            ranking = self._filter_seen(playlist_id, ranking)

        return ranking[:at]

    def _filter_seen(self, playlist_id, ranking):
        songs_playlist = self.URM_train[playlist_id]
        seen = songs_playlist.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]


    def similarityMatrixTopK(item_weights, forceSparseOutput=True, k=100, verbose=False, inplace=True):
        """
        The function selects the TopK most similar elements, column-wise

        :param item_weights:
        :param forceSparseOutput:
        :param k:
        :param verbose:
        :param inplace: Default True, WARNING matrix will be modified
        :return:
        """

        assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

        start_time = time.time()

        if verbose:
            print("Generating topK matrix")

        nitems = item_weights.shape[1]
        k = min(k, nitems)

        # for each column, keep only the top-k scored items
        sparse_weights = not isinstance(item_weights, np.ndarray)

        if not sparse_weights:

            idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

            if inplace:
                W = item_weights
            else:
                W = item_weights.copy()

            # index of the items that don't belong to the top-k similar items of each column
            not_top_k = idx_sorted[:-k, :]
            # use numpy fancy indexing to zero-out the values in sim without using a for loop
            W[not_top_k, np.arange(nitems)] = 0.0

            if forceSparseOutput:
                W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

                if verbose:
                    print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

                return W_sparse

            if verbose:
                print("Dense TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W

        else:
            # iterate over each column and keep only the top-k similar items
            data, rows_indices, cols_indptr = [], [], []

            item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

            for item_idx in range(nitems):
                cols_indptr.append(len(data))

                start_position = item_weights.indptr[item_idx]
                end_position = item_weights.indptr[item_idx + 1]

                column_data = item_weights.data[start_position:end_position]
                column_row_index = item_weights.indices[start_position:end_position]

                idx_sorted = np.argsort(column_data)  # sort by column
                top_k_idx = idx_sorted[-k:]

                data.extend(column_data[top_k_idx])
                rows_indices.extend(column_row_index[top_k_idx])

            cols_indptr.append(len(data))

            # During testing CSR is faster
            W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
            W_sparse = W_sparse.tocsr()

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse
'''
