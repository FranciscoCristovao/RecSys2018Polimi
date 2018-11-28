#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""
import multiprocessing

from utils.auxUtils import check_matrix, filter_seen, buildURMMatrix, similarityMatrixTopK

import subprocess
import os, sys, time, platform

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Evaluator
# from slimRS.recommender import Recommender


# class SLIM_BPR_Cython(Similarity_Matrix_Recommender, Recommender):
class SLIM_BPR_Cython():

    def __init__(self, train_data, URM_validation = None,
                 recompile_cython=True, final_model_sparse_weights=True, train_with_sparse_weights=False,
                 symmetric=True):

        # super(SLIM_BPR_Cython, self).__init__()

        self.URM_train = check_matrix(buildURMMatrix(train_data), 'csr')
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values

        self.n_users = self.URM_train.shape[0]
        self.n_items = self.URM_train.shape[1]
        self.normalize = False

        self.train_with_sparse_weights = train_with_sparse_weights
        self.sparse_weights = final_model_sparse_weights

        if URM_validation is not None:
            self.URM_validation = URM_validation.copy()
        else:
            self.URM_validation = None

        if self.train_with_sparse_weights:
            self.sparse_weights = True

        self.URM_mask = self.URM_train.copy()

        self.URM_mask.eliminate_zeros()

        self.symmetric = symmetric

        if not self.train_with_sparse_weights:

            n_items = self.URM_train.shape[1]
            requiredGB = 8 * n_items**2 / 1e+06

            if symmetric:
                requiredGB /= 2

            print("SLIM_BPR_Cython: Estimated memory required for similarity matrix of {} items is {:.2f} MB".format(n_items, requiredGB))

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def fit(self, epochs=160, logFile=None, playlist_ids=None, filterTopPop=False,
            batch_size=1000, lambda_i=0.001, lambda_j=0.0001, learning_rate=0.001, topK=200,
            sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999,
            stop_on_validation=False, lower_validatons_allowed=10, validation_metric="map",
            validation_function=None, validation_every_n=10):
        '''
        :param epochs:
        :param filterTopPop:
        :param batch_size:
        :param lambda_i: parameter for weighting the SLIM, proposed by paper: 0.0025
        :param lambda_j: parameter for weighting the SLIM, proposed by paper: 0.00025
        :param learning_rate: how much the algorithm is learning for each epoch
        :param topK: knn similarity
        :param sgd_mode: adagrad, rmsprop, adam, sgd
        :param gamma: rmsprop value
        :param beta_1: adam value proposed by paper: 0.9
        :param beta_2: adam value proposed by paper: 0.999
        '''

        print('Fitting..')
        # Import compiled module
        from slimRS.Cython.SLIM_BPR_Cython_Epoch import SLIM_BPR_Cython_Epoch
        print('Cython module imported')
        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()
        URM_train_positive.eliminate_zeros()

        self.sgd_mode = sgd_mode
        self.epochs = epochs
        self.cythonEpoch = SLIM_BPR_Cython_Epoch(self.URM_mask,
                                                 train_with_sparse_weights = self.train_with_sparse_weights,
                                                 final_model_sparse_weights = self.sparse_weights,
                                                 topK=topK,
                                                 learning_rate=learning_rate,
                                                 li_reg = lambda_i,
                                                 lj_reg = lambda_j,
                                                 batch_size=1,
                                                 symmetric = self.symmetric,
                                                 sgd_mode = sgd_mode,
                                                 gamma=gamma,
                                                 beta_1=beta_1,
                                                 beta_2=beta_2)

        if(topK != False and topK<1):
            raise ValueError("TopK not valid. Acceptable values are either False or a positive integer value. Provided value was '{}'".format(topK))
        self.topK = topK

        self.logFile = logFile

        if validation_every_n is not None:
            self.validation_every_n = validation_every_n
        else:
            self.validation_every_n = np.inf

        if validation_function is None:
            validation_function = self.default_validation_function

        print('After validation')

        self.batch_size = batch_size
        self.lambda_i = lambda_i
        self.lambda_j = lambda_j
        self.learning_rate = learning_rate


        start_time = time.time()

        print('Time has started')
        best_validation_metric = None
        lower_validatons_count = 0
        convergence = False

        self.S_incremental = self.cythonEpoch.get_S()
        self.S_best = self.S_incremental.copy()
        self.epochs_best = 0

        currentEpoch = 0

        while currentEpoch < self.epochs and not convergence:

            if self.batch_size>0:
                self.cythonEpoch.epochIteration_Cython()
            else:
                print("No batch not available")

            # Determine whether a validaton step is required
            if self.URM_validation is not None and (currentEpoch + 1) % self.validation_every_n == 0:

                print("SLIM_BPR_Cython: Validation begins...")

                self.get_S_incremental_and_set_W()

                results_run = validation_function(playlist_ids)

                print("SLIM_BPR_Cython: {}".format(results_run))

                # Update the D_best and V_best
                # If validation is required, check whether result is better
                if stop_on_validation:

                    current_metric_value = results_run  # results_run[validation_metric]

                    if best_validation_metric is None or best_validation_metric < current_metric_value:

                        best_validation_metric = current_metric_value
                        self.S_best = self.S_incremental.copy()
                        self.epochs_best = currentEpoch +1
                        lower_validatons_count = 0

                    else:
                        lower_validatons_count += 1

                    if lower_validatons_count >= lower_validatons_allowed:
                        convergence = True
                        print("SLIM_BPR_Cython: Convergence reached! Terminating at epoch {}. Best value for '{}' at epoch {} is {:.4f}. Elapsed time {:.2f} min".format(
                            currentEpoch+1, validation_metric, self.epochs_best, best_validation_metric, (time.time() - start_time) / 60))

            # If no validation required, always keep the latest
            if not stop_on_validation:
                self.S_best = self.S_incremental.copy()

            print("SLIM_BPR_Cython: Epoch {} of {}. Elapsed time {:.2f} min".format(
                currentEpoch+1, self.epochs, (time.time() - start_time) / 60))

            currentEpoch += 1

        self.get_S_incremental_and_set_W()
        print('Finishing...')
        sys.stdout.flush()

    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'lambda_i': self.lambda_i,
                          'lambda_j': self.lambda_j,
                          'batch_size': self.batch_size,
                          'learn_rate': self.learning_rate,
                          'topK_similarity': self.topK,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))
        # print("Weights: {}\n".format(str(list(self.weights))))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            # logFile.write("Weights: {}\n".format(str(list(self.weights))))
            logFile.flush()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/slimRS/Cython"
        #fileToCompile_list = ['Sparse_Matrix_CSR.pyx', 'SLIM_BPR_Cython_Epoch.pyx']
        fileToCompile_list = ['SLIM_BPR_Cython_Epoch.pyx']

        for fileToCompile in fileToCompile_list:
            if platform.system() == 'Windows':
                cmd = 'python'
            else:
                cmd = 'python3'

            command = [cmd,
                       'compileCython.py',
                       fileToCompile,
                       'build_ext',
                       '--inplace'
                       ]

            output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            try:

                command = ['cython',
                           fileToCompile,
                           '-a'
                           ]

                output = subprocess.check_output(' '.join(command), shell=True, cwd=os.getcwd() + compiledModuleSubfolder)

            except:
                pass


        print("Compiled module saved in subfolder: {}".format(compiledModuleSubfolder))

        # Command to run compilation script
        # python compileCython.py SLIM_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        # cython -a SLIM_BPR_Cython_Epoch.pyx


    def get_S_incremental_and_set_W(self):

        self.S_incremental = self.cythonEpoch.get_S()

        if self.train_with_sparse_weights:
            self.W_sparse = self.S_incremental
        else:
            if self.sparse_weights:
                self.W_sparse = similarityMatrixTopK(self.S_incremental, k=self.topK)
            else:
                self.W = self.S_incremental

    def get_weight_matrix(self):
        if self.train_with_sparse_weights:
            matrix_w = self.W_sparse
        else:
            if self.sparse_weights:
                matrix_w = self.W_sparse
            else:
                matrix_w = self.W
        return csr_matrix(matrix_w, shape=(self.n_items, self.n_items))

    def get_estimated_ratings(self):
        matrix_W = self.get_weight_matrix()
        return check_matrix(self.URM_train.dot(matrix_W), 'csr')

    def recommend(self, playlist_ids):

        print("Recommending...")

        final_prediction = {}

        if self.train_with_sparse_weights:
            matrix_W = self.W_sparse
        else:
            if self.sparse_weights:
                matrix_W = self.W_sparse
            else:
                matrix_W = self.W

        # what dimension does W have?
        self.W = csr_matrix(matrix_W, shape=(self.n_items, self.n_items))
        estimated_ratings = check_matrix(self.URM_train.dot(self.W), 'csr')

        counter = 0

        for k in playlist_ids:

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

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def default_validation_function(self, playlist_ids):
        e = Evaluator()
        return e.evaluate(self.recommend(playlist_ids), self.URM_validation)

    def evaluateRecommendations(self, URM_test, at=10, minRatingsPerUser=1, exclude_seen=True,
                                mode='parallel', filterTopPop = False,
                                filterCustomItems = np.array([], dtype=np.int),
                                filterCustomUsers = np.array([], dtype=np.int)):
        """
        Speed info:
        - Sparse weights: batch mode is 2x faster than sequential
        - Dense weights: batch and sequential speed are equivalent


        :param URM_test_new:            URM to be used for testing
        :param at: 10                   Length of the recommended items
        :param minRatingsPerUser: 1     Users with less than this number of interactions will not be evaluated
        :param exclude_seen: True       Whether to remove already seen items from the recommended items

        :param mode: 'sequential', 'parallel', 'batch'
        :param filterTopPop: False or decimal number        Percentage of items to be removed from recommended list and testing interactions
        :param filterCustomItems: Array, default empty           Items ID to NOT take into account when recommending
        :param filterCustomUsers: Array, default empty           Users ID to NOT take into account when recommending
        :return:
        """

        if len(filterCustomItems) == 0:
            self.filterCustomItems = False
        else:
            self.filterCustomItems = True
            self.filterCustomItems_ItemsID = np.array(filterCustomItems)

        '''
        if filterTopPop != False:

            self.filterTopPop = True

            _,_, self.filterTopPop_ItemsID = removeTopPop(self.URM_train, URM_2 = URM_test_new, percentageToRemove=filterTopPop)

            print("Filtering {}% TopPop items, count is: {}".format(filterTopPop*100, len(self.filterTopPop_ItemsID)))

            # Zero-out the items in order to be considered irrelevant
            URM_test_new = check_matrix(URM_test_new, format='lil')
            URM_test_new[:,self.filterTopPop_ItemsID] = 0
            URM_test_new = check_matrix(URM_test_new, format='csr')

        '''

        # During testing CSR is faster
        self.URM_test = check_matrix(URM_test, format='csr')
        self.evaluator = Evaluator()
        self.URM_train = check_matrix(self.URM_train, format='csr')
        self.at = at
        self.minRatingsPerUser = minRatingsPerUser
        self.exclude_seen = exclude_seen


        nusers = self.URM_test.shape[0]

        # Prune users with an insufficient number of ratings
        rows = self.URM_test.indptr
        numRatings = np.ediff1d(rows)
        mask = numRatings >= minRatingsPerUser
        usersToEvaluate = np.arange(nusers)[mask]

        if len(filterCustomUsers) != 0:
            print("Filtering {} Users".format(len(filterCustomUsers)))
            usersToEvaluate = set(usersToEvaluate) - set(filterCustomUsers)

        usersToEvaluate = list(usersToEvaluate)



        if mode=='sequential':
            return self.evaluateRecommendationsSequential(usersToEvaluate)
        elif mode=='parallel':
            return self.evaluateRecommendationsParallel(usersToEvaluate)
        elif mode=='batch':
            return self.evaluateRecommendationsBatch(usersToEvaluate)
        elif mode=='cython':
             return self.evaluateRecommendationsCython(usersToEvaluate)
        # elif mode=='random-equivalent':
        #     return self.evaluateRecommendationsRandomEquivalent(usersToEvaluate)
        else:
            raise ValueError("Mode '{}' not available".format(mode))

    def evaluateOneUser(self, test_user):

        # Being the URM CSR, the indices are the non-zero column indexes
        # relevant_items = self.URM_test_relevantItems[test_user]
        relevant_items = self.URM_test[test_user].indices

        # this will rank top n items
        recommended_items = self.recommend(test_user)

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # evaluate the recommendation list with ranking metrics ONLY

        map_ = self.evaluator.map(is_relevant, relevant_items)

        return map_

    def evaluateRecommendationsParallel(self, usersToEvaluate):

        print("Evaluation of {} users begins".format(len(usersToEvaluate)))

        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count(), maxtasksperchild=1)
        resultList = pool.map(self.evaluateOneUser, usersToEvaluate)

        # for i, _ in enumerate(pool.imap_unordered(self.evaluateOneUser, usersToEvaluate), 1):
        #    if(i%1000 == 0):
        #        sys.stderr.write('\rEvaluated {} users ({0:%})'.format(i , i / usersToEvaluate))

        # Close the pool to avoid memory leaks
        pool.close()

        n_eval = len(usersToEvaluate)
        map_= 0.0

        # Looping is slightly faster then using the numpy vectorized approach, less data transformation
        for result in resultList:
            map_ += result[0]

        if (n_eval > 0):

            map_ /= n_eval


        else:
            print("WARNING: No users had a sufficient number of relevant items")

        results_run = {}

        results_run["map"] = map_

        return (results_run)
