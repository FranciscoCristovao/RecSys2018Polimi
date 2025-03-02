"""
Created on 07/09/17
@author: Maurizio Ferrari Dacrema
"""

import subprocess
import os, sys
import time
import numpy as np
from utils.auxUtils import buildURMMatrix
from matrixFactorizationRS.Cython.MF_BPR_Cython_Epoch import MF_BPR_Cython_Epoch
import platform


class MF_BPR_Cython():


    def __init__(self, train_data, recompile_cython = False,
                 num_factors=200):

        self.URM_train = buildURMMatrix(train_data)
        self.n_users = self.URM_train.shape[0]
        self.n_items = self.URM_train.shape[1]
        self.normalize = False
        self.num_factors = num_factors

        if recompile_cython:
            print("Compiling in Cython")
            self.runCompilationScript()
            print("Compilation Complete")

    def fit(self, epochs=200, logFile=None, URM_test=None, filterTopPop = False, filterCustomItems = np.array([], dtype=np.int), minRatingsPerUser=1,
            batch_size=1000, validate_every_N_epochs=1, start_validation_after_N_epochs=0,
            learning_rate=0.01, sgd_mode='sgd', user_reg=0.0, positive_reg=0.0, negative_reg=0.0):


        self.eligibleUsers = []

        # Select only positive interactions
        URM_train_positive = self.URM_train.copy()

        URM_train_positive.eliminate_zeros()


        for user_id in range(self.n_users):

            start_pos = URM_train_positive.indptr[user_id]
            end_pos = URM_train_positive.indptr[user_id+1]

            numUserInteractions = len(URM_train_positive.indices[start_pos:end_pos])

            if  numUserInteractions > 0 and numUserInteractions<self.n_items:
                self.eligibleUsers.append(user_id)

        # self.eligibleUsers contains the userID having at least one positive interaction and one item non observed
        self.eligibleUsers = np.array(self.eligibleUsers, dtype=np.int64)
        self.sgd_mode = sgd_mode

        self.cythonEpoch = MF_BPR_Cython_Epoch(URM_train_positive,
                                                 self.eligibleUsers,
                                                 num_factors=self.num_factors,
                                                 learning_rate=learning_rate,
                                                 batch_size=1,
                                                 sgd_mode=sgd_mode,
                                                 user_reg=user_reg,
                                                 positive_reg=positive_reg,
                                                 negative_reg=negative_reg)

        self.batch_size = batch_size
        self.learning_rate = learning_rate

        start_time_train = time.time()

        for currentEpoch in range(epochs):

            start_time_epoch = time.time()

            if currentEpoch > 0:
                if self.batch_size>0:
                    self.epochIteration()
                else:
                    print("No batch not available")

            if (URM_test is not None) and (currentEpoch % validate_every_N_epochs == 0) and \
                            currentEpoch >= start_validation_after_N_epochs:

                print("Evaluation begins")

                self.W = self.cythonEpoch.get_W()
                self.H = self.cythonEpoch.get_H()

                results_run = self.evaluateRecommendations(URM_test,
                                                           minRatingsPerUser=minRatingsPerUser)

                self.writeCurrentConfig(currentEpoch, results_run, logFile)

                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                     float(time.time() - start_time_epoch) / 60))


            # Fit with no validation
            else:
                print("Epoch {} of {} complete in {:.2f} minutes".format(currentEpoch, epochs,
                                                                         float(time.time() - start_time_epoch) / 60))

        # Ensure W and H are up to date
        self.W = self.cythonEpoch.get_W()
        self.H = self.cythonEpoch.get_H()

        print("Fit completed in {:.2f} minutes".format(float(time.time() - start_time_train) / 60))

        sys.stdout.flush()

    def runCompilationScript(self):

        # Run compile script setting the working directory to ensure the compiled file are contained in the
        # appropriate subfolder and not the project root

        compiledModuleSubfolder = "/MatrixFactorization/Cython"
        fileToCompile_list = ['MF_BPR_Cython_Epoch.pyx']

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
        #python compileCython.py MF_BPR_Cython_Epoch.pyx build_ext --inplace

        # Command to generate html report
        #subprocess.call(["cython", "-a", "MF_BPR_Cython_Epoch.pyx"])

    def epochIteration(self):

        self.cythonEpoch.epochIteration_Cython()

    def writeCurrentConfig(self, currentEpoch, results_run, logFile):

        current_config = {'learn_rate': self.learning_rate,
                          'num_factors': self.num_factors,
                          'batch_size': 1,
                          'epoch': currentEpoch}

        print("Test case: {}\nResults {}\n".format(current_config, results_run))

        sys.stdout.flush()

        if (logFile != None):
            logFile.write("Test case: {}, Results {}\n".format(current_config, results_run))
            logFile.flush()

    def recommend(self, user_id, n=10, exclude_seen=True, filterTopPop = False, filterCustomItems = False):

        # compute the scores using the dot product
        user_profile = self.URM_train[user_id]

        scores_array = np.dot(self.W[user_id], self.H.T)


        if self.normalize:
            # normalization will keep the scores in the same range
            # of value of the ratings in dataset
            rated = user_profile.copy()
            rated.data = np.ones_like(rated.data)
            if self.sparse_weights:
                den = rated.dot(self.W_sparse).toarray().ravel()
            else:
                den = rated.dot(self.W).ravel()
            den[np.abs(den) < 1e-6] = 1.0  # to avoid NaNs
            scores_array /= den

        if exclude_seen:
            scores_array = self._filter_seen_on_scores(user_id, scores_array)
        '''
        if filterTopPop:
            scores_array = self._filter_TopPop_on_scores(scores_array)

        if filterCustomItems:
            scores_array = self._filterCustomItems_on_scores(scores_array)
        '''

        # rank items and mirror column to obtain a ranking in descending score
        #ranking = scores.argsort()
        #ranking = np.flip(ranking, axis=0)

        # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
        # - Partition the data to extract the set of relevant items
        # - Sort only the relevant items
        # - Get the original item index
        relevant_items_partition = (-scores_array).argpartition(n)[0:n]
        relevant_items_partition_sorting = np.argsort(-scores_array[relevant_items_partition])
        ranking = relevant_items_partition[relevant_items_partition_sorting]

        return ranking

    def _filter_seen_on_scores(self, user_id, scores):

        seen = self.URM_train.indices[self.URM_train.indptr[user_id]:self.URM_train.indptr[user_id + 1]]

        scores[seen] = -np.inf

        return scores
