"""
Created on 07/09/17

@author: Maurizio Ferrari Dacrema
"""

#cython: boundscheck=False
#cython: wraparound=False
#cython: initializedcheck=False
#cython: language_level=3
#cython: nonecheck=False
#cython: cdivision=True
#cython: unpack_method_calls=True
#cython: overflowcheck=False

#defining NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from Base.Recommender_utils import check_matrix
import numpy as np
cimport numpy as np
import time
import sys

from libc.math cimport exp, sqrt
from libc.stdlib cimport rand, RAND_MAX


cdef struct BPR_sample:
    long user
    long pos_item
    long neg_item


cdef class MF_BPR_Cython_Epoch:

    cdef int n_users, n_items, n_factors
    cdef int numPositiveIteractions

    cdef float learning_rate, user_reg, positive_reg, negative_reg

    cdef int batch_size

    cdef int[:] URM_mask_indices, URM_mask_indptr

    cdef double[:,:] W, H


    # Adaptive gradient
    cdef int useAdaGrad, useRmsprop, useAdam

    cdef double [:] sgd_cache_I, sgd_cache_U
    cdef double gamma

    cdef double [:] sgd_cache_I_momentum_1, sgd_cache_I_momentum_2
    cdef double [:] sgd_cache_U_momentum_1, sgd_cache_U_momentum_2
    cdef double beta_1, beta_2, beta_1_power_t, beta_2_power_t
    cdef double momentum_1, momentum_2


    def __init__(self, URM_mask, n_factors = 10,
                 learning_rate = 0.01, user_reg = 0.0, positive_reg = 0.0, negative_reg = 0.0,
                 batch_size = 1, sgd_mode='sgd', gamma=0.995, beta_1=0.9, beta_2=0.999):

        super(MF_BPR_Cython_Epoch, self).__init__()


        URM_mask = check_matrix(URM_mask, 'csr')

        self.numPositiveIteractions = int(URM_mask.nnz * 1)
        self.n_users = URM_mask.shape[0]
        self.n_items = URM_mask.shape[1]
        self.n_factors = n_factors

        self.URM_mask_indices = URM_mask.indices
        self.URM_mask_indptr = URM_mask.indptr

        # W and H cannot be initialized as zero, otherwise the gradient will always be zero
        self.W = np.random.random((self.n_users, self.n_factors))
        self.H = np.random.random((self.n_items, self.n_factors))



        self.useAdaGrad = False
        self.useRmsprop = False
        self.useAdam = False


        if sgd_mode=='adagrad':
            self.useAdaGrad = True
            self.sgd_cache_I = np.zeros((self.n_items), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.n_users), dtype=np.float64)

        elif sgd_mode=='rmsprop':
            self.useRmsprop = True
            self.sgd_cache_I = np.zeros((self.n_items), dtype=np.float64)
            self.sgd_cache_U = np.zeros((self.n_users), dtype=np.float64)

            # Gamma default value suggested by Hinton
            # self.gamma = 0.9
            self.gamma = gamma

        elif sgd_mode=='adam':
            self.useAdam = True
            self.sgd_cache_I_momentum_1 = np.zeros((self.n_items), dtype=np.float64)
            self.sgd_cache_I_momentum_2 = np.zeros((self.n_items), dtype=np.float64)

            self.sgd_cache_U_momentum_1 = np.zeros((self.n_users), dtype=np.float64)
            self.sgd_cache_U_momentum_2 = np.zeros((self.n_users), dtype=np.float64)

            # Default value suggested by the original paper
            # beta_1=0.9, beta_2=0.999
            self.beta_1 = beta_1
            self.beta_2 = beta_2
            self.beta_1_power_t = beta_1
            self.beta_2_power_t = beta_2

        elif sgd_mode=='sgd':
            pass
        else:
            raise ValueError(
                "SGD_mode not valid. Acceptable values are: 'sgd', 'adagrad', 'rmsprop', 'adam'. Provided value was '{}'".format(
                    sgd_mode))



        self.learning_rate = learning_rate
        self.user_reg = user_reg
        self.positive_reg = positive_reg
        self.negative_reg = negative_reg


        if batch_size!=1:
            print("MiniBatch not implemented, reverting to default value 1")
        self.batch_size = 1


    # Using memoryview instead of the sparse matrix itself allows for much faster access
    cdef int[:] getSeenItems(self, long index):
        return self.URM_mask_indices[self.URM_mask_indptr[index]:self.URM_mask_indptr[index + 1]]



    def epochIteration_Cython(self):

        # Get number of available interactions
        cdef long totalNumberOfBatch = int(self.n_users / self.batch_size) + 1


        cdef BPR_sample sample
        cdef long u, i, j
        cdef long index, numCurrentBatch
        cdef double x_uij, sigmoid_user, sigmoid_item

        cdef int numSeenItems


        cdef double H_i, H_j, W_u


        cdef long start_time_epoch = time.time()
        cdef long start_time_batch = time.time()

        for numCurrentBatch in range(totalNumberOfBatch):

            # Uniform user sampling with replacement
            sample = self.sampleBPR_Cython()

            u = sample.user
            i = sample.pos_item
            j = sample.neg_item

            x_uij = 0.0

            for index in range(self.n_factors):

                x_uij = self.W[u,index] * (self.H[i,index] - self.H[j,index])

            # Use gradient of log(sigm(-x_uij))
            sigmoid_item = 1 / (1 + exp(x_uij))
            sigmoid_user = sigmoid_item

            #print(sample)

            sigmoid_item_i = self.adaptive_gradient_item(sigmoid_item, i)
            sigmoid_item_j = self.adaptive_gradient_item(sigmoid_item, j)

            sigmoid_user = self.adaptive_gradient_user(sigmoid_user, u)




            for index in range(self.n_factors):

                # Copy original value to avoid messing up the updates
                H_i = self.H[i, index]
                H_j = self.H[j, index]
                W_u = self.W[u, index]

                self.W[u, index] += self.learning_rate * (sigmoid_user * ( H_i - H_j ) - self.user_reg * W_u)
                self.H[i, index] += self.learning_rate * (sigmoid_item_i * ( W_u ) - self.positive_reg * H_i)
                self.H[j, index] += self.learning_rate * (sigmoid_item_j * (-W_u ) - self.negative_reg * H_j)

            #input()

            if((numCurrentBatch%5000000==0 and not numCurrentBatch==0) or numCurrentBatch==totalNumberOfBatch-1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    numCurrentBatch*self.batch_size,
                    100.0* float(numCurrentBatch*self.batch_size)/self.n_users,
                    time.time() - start_time_batch,
                    float(numCurrentBatch*self.batch_size + 1) / (time.time() - start_time_epoch)))

                sys.stdout.flush()
                sys.stderr.flush()

                start_time_batch = time.time()


    def get_W(self):
        return np.array(self.W)


    def get_H(self):
        return np.array(self.H)



    cdef double adaptive_gradient_item(self, double gradient, long item_id):


        cdef double gradient_update


        if self.useAdaGrad:
            self.sgd_cache_I[item_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)


        elif self.useRmsprop:
            self.sgd_cache_I[item_id] = self.sgd_cache_I[item_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_I[item_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_I_momentum_1[item_id] = \
                self.sgd_cache_I_momentum_1[item_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_I_momentum_2[item_id] = \
                self.sgd_cache_I_momentum_2[item_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_I_momentum_1[item_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_I_momentum_2[item_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient


        return gradient_update



    cdef double adaptive_gradient_user(self, double gradient, long user_id):


        cdef double gradient_update

        if self.useAdaGrad:
            self.sgd_cache_U[user_id] += gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)


        elif self.useRmsprop:
            self.sgd_cache_U[user_id] = self.sgd_cache_U[user_id] * self.gamma + (1 - self.gamma) * gradient ** 2

            gradient_update = gradient / (sqrt(self.sgd_cache_U[user_id]) + 1e-8)


        elif self.useAdam:

            self.sgd_cache_U_momentum_1[user_id] = \
                self.sgd_cache_U_momentum_1[user_id] * self.beta_1 + (1 - self.beta_1) * gradient

            self.sgd_cache_U_momentum_2[user_id] = \
                self.sgd_cache_U_momentum_2[user_id] * self.beta_2 + (1 - self.beta_2) * gradient**2


            self.momentum_1 = self.sgd_cache_U_momentum_1[user_id]/ (1 - self.beta_1_power_t)
            self.momentum_2 = self.sgd_cache_U_momentum_2[user_id]/ (1 - self.beta_2_power_t)

            gradient_update = self.momentum_1/ (sqrt(self.momentum_2) + 1e-8)


        else:

            gradient_update = gradient


        return gradient_update





    cdef BPR_sample sampleBPR_Cython(self):

        cdef BPR_sample sample = BPR_sample(-1,-1,-1)
        cdef long index, start_pos_seen_items, end_pos_seen_items

        cdef int negItemSelected, numSeenItems = 0


        # Skip users with no interactions or with no negative items
        while numSeenItems == 0 or numSeenItems == self.n_items:

            sample.user = rand() % self.n_users

            start_pos_seen_items = self.URM_mask_indptr[sample.user]
            end_pos_seen_items = self.URM_mask_indptr[sample.user+1]

            numSeenItems = end_pos_seen_items - start_pos_seen_items


        index = rand() % numSeenItems

        sample.pos_item = self.URM_mask_indices[start_pos_seen_items + index]



        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        # for every user
        while (not negItemSelected):

            sample.neg_item = rand() % self.n_items

            index = 0
            while index < numSeenItems and self.URM_mask_indices[start_pos_seen_items + index]!=sample.neg_item:
                index+=1

            if index == numSeenItems:
                negItemSelected = True


        return sample
