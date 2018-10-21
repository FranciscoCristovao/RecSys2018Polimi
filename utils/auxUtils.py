import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import scipy

class Helper:

    def __init__(self):
        print("Helper has been initialized")

    def buildURMMatrix(self, data):

        playlists = data["playlist_id"].values
        tracks = data["track_id"].values
        interaction = np.ones(len(tracks))
        coo_urm = coo_matrix((interaction, (playlists, tracks)))
        # print("This is the coo_urm", coo_urm)
        return coo_urm.tocsr()

    def buildICMMatrix(self, data):

        '''
        frames = [pd.get_dummies(data['album_id']), pd.get_dummies(data['artist_id'])]
        aux = pd.concat(frames, axis=1)
        return csr_matrix(aux.values)'''

        '''
        frames = [pd.get_dummies(data['album_id']), pd.get_dummies(data['artist_id'])]
        aux = pd.concat(frames, axis=1)
        return csr_matrix(pd.get_dummies(data['artist_id'].values)'''

        tracks = data["track_id"].values
        artists = data["artist_id"].values
        interaction = np.ones(len(tracks))
        coo_icm = coo_matrix((interaction, (tracks, artists)))
        print("Coo icm with artists correctly built")
        return coo_icm.tocsr()
        '''

        tracks = data["track_id"].values
        albums = data["album_id"].values
        artists = data["artist_id"].values
        features = np.concatenate([albums, artists])
        tracks_sized = np.concatenate([tracks, tracks])
        interaction = np.ones(len(features))
        coo_icm = coo_matrix((interaction, (tracks_sized, features)))
        return coo_icm.tocsr()'''

    def dataframeToCSR(self, data):
        print(csr_matrix(data))


class Cosine:

    def compute(self, mat, shrinkage):
        # convert to csc matrix for faster column-wise operations
        mat = mat.tocsc()
        # print(type(mat))

        # 2) compute the cosine similarity using the dot-product
        dist = mat * mat.T
        print("Computed")

        # zero out diagonal values
        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")
        '''
        #SHRINKAGE FOR LATER
        # and apply the shrinkage
        if shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")
        '''
        return csr_matrix(dist)

    def apply_shrinkage(self, icm, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        icm_ind = icm.copy()
        icm_ind.data = np.ones_like(icm_ind.data)
        # compute the co-rated counts
        co_counts = icm_ind * icm_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


class Evaluator:

    helper = Helper()

    def __init__(self):
        print("Evaluator has been initialized")
        self.helper = Helper()

    def map(self, recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

        return map_score

    def evaluate(self, recommended, test_data):
        cumulative_map = 0.0
        num_eval = 0
        counter = 0
        urm_test = self.helper.buildURMMatrix(test_data)

        for i in recommended["playlist_id"]:

            relevant_items = urm_test[i].indices

            if len(relevant_items) > 0:
                recommended_items = np.fromstring(recommended["track_ids"][counter], dtype=int, sep=' ')
                num_eval += 1

                cumulative_map += self.map(recommended_items, relevant_items)

            counter += 1

        cumulative_map /= num_eval
        print("Evaluated", num_eval, "playlists")

        print("Recommender performance is: MAP = {:.4f}".format(cumulative_map))
