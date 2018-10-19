import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

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


class Evaluator:

    helper = Helper()

    def __init__(self):
        print("Evaluator has been initialized")
        self.helper = Helper()

    def map(self, recommended_items, relevant_items):
        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        # Cumulative sum: precision at 1, at 2, at 3 ...
        p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

        if(np.min([relevant_items.shape[0], is_relevant.shape[0]]) > 0):
            map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        else:
            map_score = 0.0
        return map_score

    def evaluate(self, recommended, test_data):
        cumulative_map = 0.0
        num_eval = 0
        urm_test = self.helper.buildURMMatrix(test_data)
        for i in recommended["playlist_id"]:
            # print("WTF is this", i)
            relevant_items = urm_test[i].indices

            if len(relevant_items) > 0:
                recommended_items = np.fromstring(recommended["track_ids"][num_eval], dtype=int, sep=' ')
                num_eval += 1

                # print("Recommendations", recommended_items)
                # print("Relevant Items", relevant_items)
                # print("MAP", self.map(recommended_items, relevant_items))
                cumulative_map += self.map(recommended_items, relevant_items)

        cumulative_map /= num_eval
        print("Evaluated", num_eval, "playlists")

        print("Recommender performance is: MAP = {:.4f}".format(cumulative_map))