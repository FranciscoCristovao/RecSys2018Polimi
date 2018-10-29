import numpy as np
import time
from utils.auxUtils import similarityMatrixTopK, buildURMMatrix

class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, data, learning_rate=0.01, epochs=3):
        print("Slim initialized...")
        self.URM = buildURMMatrix(data)
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.URM_mask = self.URM.copy()
        self.URM_mask.eliminate_zeros()

        self.n_users = self.URM_mask.shape[0]
        self.n_items = self.URM_mask.shape[1]

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.URM_mask.indptr[user_id]
            end_pos = self.URM_mask.indptr[user_id + 1]

            if len(self.URM_mask.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        userSeenItems = self.URM_mask[user_id, :].indices
        pos_item_id = np.random.choice(userSeenItems)

        negItemSelected = False

        # It's faster to just try again then to build a mapping of the non-seen items
        while (not negItemSelected):
            neg_item_id = np.random.randint(0, self.n_items)

            if (neg_item_id not in userSeenItems):
                negItemSelected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):
        print("Starting new epoch")
        # Get number of available interactions
        numPositiveIteractions = int(self.URM_mask.nnz)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(numPositiveIteractions):
            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            userSeenItems = self.URM_mask[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, userSeenItems].sum()
            x_j = self.similarity_matrix[negative_item_id, userSeenItems].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update
            self.similarity_matrix[positive_item_id, userSeenItems] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, userSeenItems] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == numPositiveIteractions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / numPositiveIteractions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self):
        print("Fitting...")
        for numEpoch in range(self.epochs):
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=100)

    def recommend(self, user_id, at=None, exclude_seen=True):

        # compute the scores using the dot product
        user_profile = self.URM[user_id]
        scores = user_profile.dot(self.similarity_matrix)
        scores = scores.toarray()

        # rank items
        ranking = scores.argsort()[::-1].squeeze()
        if exclude_seen:
            ranking = self._filter_seen(user_id, ranking)

        return ranking[:at]

    def _filter_seen(self, user_id, ranking):
        user_profile = self.URM[user_id]
        seen = user_profile.indices
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]