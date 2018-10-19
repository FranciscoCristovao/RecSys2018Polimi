import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix

class Helper:

    def __init__(self):
        print("Helper has been initialized")

    def buildURMMatrix(self, data):

        playlists = data["playlist_id"].values
        tracks = data["track_id"].values
        interaction = np.ones(len(playlists))
        coo_urm = coo_matrix((interaction, (playlists, tracks)))
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
