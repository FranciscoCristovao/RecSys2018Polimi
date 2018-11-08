import numpy as np
import time
from utils.auxUtils import similarityMatrixTopK, buildURMMatrix


class SLIM_BPR_Recommender(object):
    """ SLIM_BPR recommender with cosine similarity and no shrinkage"""

    def __init__(self, data, learning_rate=0.05, epochs=2):
        print("Slim initialized...")
        self.URM = buildURMMatrix(data)
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.interactions = self.URM.copy()
        self.interactions.eliminate_zeros()

        self.n_users = self.interactions.shape[0]
        self.n_items = self.interactions.shape[1]

        self.similarity_matrix = np.zeros((self.n_items, self.n_items))

        # Extract users having at least one interaction to choose from
        self.eligibleUsers = []

        for user_id in range(self.n_users):

            start_pos = self.interactions.indptr[user_id]
            end_pos = self.interactions.indptr[user_id + 1]

            if len(self.interactions.indices[start_pos:end_pos]) > 0:
                self.eligibleUsers.append(user_id)

    def sampleTriplet(self):

        # By randomly selecting a user in this way we could end up
        # with a user with no interactions
        # user_id = np.random.randint(0, n_users)

        user_id = np.random.choice(self.eligibleUsers)

        # Get user seen items and choose one
        user_seen_items = self.interactions[user_id, :].indices
        pos_item_id = np.random.choice(user_seen_items)

        neg_item_selected = False

        # Try until we find an item not selected by the user

        while not neg_item_selected:
            neg_item_id = np.random.randint(0, self.n_items)

            if neg_item_id not in user_seen_items:
                neg_item_selected = True

        return user_id, pos_item_id, neg_item_id

    def epochIteration(self):

        # Get number of available interactions
        num_positive_interactions = int(self.interactions.nnz*0.01)

        start_time_epoch = time.time()
        start_time_batch = time.time()

        # Uniform user sampling without replacement
        for num_sample in range(num_positive_interactions):
            # Sample
            user_id, positive_item_id, negative_item_id = self.sampleTriplet()

            user_seen_items = self.interactions[user_id, :].indices

            # Prediction
            x_i = self.similarity_matrix[positive_item_id, user_seen_items].sum()
            x_j = self.similarity_matrix[negative_item_id, user_seen_items].sum()

            # Gradient
            x_ij = x_i - x_j

            gradient = 1 / (1 + np.exp(x_ij))

            # Update

            self.similarity_matrix[positive_item_id, user_seen_items] += self.learning_rate * gradient
            self.similarity_matrix[positive_item_id, positive_item_id] = 0

            self.similarity_matrix[negative_item_id, user_seen_items] -= self.learning_rate * gradient
            self.similarity_matrix[negative_item_id, negative_item_id] = 0

            if (time.time() - start_time_batch >= 30 or num_sample == num_positive_interactions - 1):
                print("Processed {} ( {:.2f}% ) in {:.2f} seconds. Sample per second: {:.0f}".format(
                    num_sample,
                    100.0 * float(num_sample) / num_positive_interactions,
                    time.time() - start_time_batch,
                    float(num_sample) / (time.time() - start_time_epoch)))

                start_time_batch = time.time()

    def fit(self):
        print("Fitting...")
        for numEpoch in range(self.epochs):
            print("STARTING EPOCH: ", numEpoch+1)
            self.epochIteration()

        self.similarity_matrix = self.similarity_matrix.T

        self.similarity_matrix = similarityMatrixTopK(self.similarity_matrix, k=200)

    def recommend(self, user_id, at=10, exclude_seen=True):

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
