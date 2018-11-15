import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from utils.auxUtils import check_matrix, filter_seen, buildICMMatrix, buildURMMatrix, normalize_tf_idf
from utils.Cython.Cosine_Similarity_Max import Cosine_Similarity
# from utils.cosine_similarity import Compute_Similarity_Python
from sklearn.preprocessing import normalize


class CbfRS:

    train_data = pd.DataFrame()

    def __init__(self, tracks_data, at, k=10, shrinkage=10, similarity='cosine', tf_idf=True,
                 weight_album=3, weight_artist=1, use_duration=False, weight_duration=0, num_clusters_duration=3):

        self.k = k
        self.at = at
        self.shrinkage = shrinkage
        self.similarity_name = similarity
        self.tf_idf = tf_idf

        # data = tracks_data.drop(columns="duration_sec")
        # len should be faster, according to this:
        #  https://stackoverflow.com/questions/15943769/how-do-i-get-the-row-count-of-a-pandas-dataframe
        self.num_album = len(tracks_data['album_id'].drop_duplicates(keep='first'))
        self.num_artists = len(tracks_data['artist_id'].drop_duplicates(keep='first'))
        # num_clusters_duration is decided a priori
        self.num_cluster_dur = num_clusters_duration
        self.use_track_duration = use_duration

        self.icm = buildICMMatrix(tracks_data, 1, 1, use_tracks_duration=self.use_track_duration)
        self.weight_album = weight_album
        self.weight_artist = weight_artist
        self.weight_dur = weight_duration

    def fit(self, train_data):
        print("Fitting...")

        if self.tf_idf:
            self.icm = normalize_tf_idf(self.icm)

        # self.icm = normalize(self.icm, norm='l2', axis=1)

        self.train_data = train_data
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values

        # calculating the row weights for the similarity...
        weights_album = np.full(self.num_album, self.weight_album)
        weights_artist = np.full(self.num_artists, self.weight_artist)
        row_weights = np.concatenate((weights_album, weights_artist), axis=0)

        if self.use_track_duration:
            weights_clust = np.full(self.num_cluster_dur, self.weight_dur)
            row_weights = np.concatenate((row_weights, weights_clust), axis=0)

        self.cosine = Cosine_Similarity(self.icm.T, self.k, self.shrinkage, normalize=True,
                                        row_weights=row_weights)
        # self.cosine = Compute_Similarity_Python(self.icm.T, self.k, self.shrinkage, normalize=True)
        self.sym = check_matrix(self.cosine.compute_similarity(), 'csr')
        self.urm = buildURMMatrix(train_data)

    def recommend(self, playlist_ids):
        print("Recommending...")

        final_prediction = {}  # pd.DataFrame([])

        print("STARTING ESTIMATION")
        # add ravel() ?
        estimated_ratings = self.get_estimated_ratings()

        counter = 0

        for k in playlist_ids:

            row = estimated_ratings.getrow(k)  # [k]

            # aux contains the indices (track_id) of the most similar songs
            indx = row.data.argsort()[::-1]
            aux = row.indices[indx]
            user_playlist = self.urm[k]

            aux = np.concatenate((aux, self.top_pop_songs), axis=None)
            top_songs = filter_seen(aux, user_playlist)[:self.at]

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if(counter % 5000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print("THEY ARE: ", abc)
        return df

    def recommend_single(self, k):
        # print("Recommending...")
        # add ravel() ?
        row = self.urm[k]
        estimated_ratings = row.dot(self.sym)  # .toarray().ravel()
        # aux = estimated_ratings.argsort()[::-1]
        indx = estimated_ratings.data.argsort()[::-1]
        aux = estimated_ratings.indices[indx]

        top_songs = filter_seen(row, aux)[:self.at]
        return top_songs

    def get_estimated_ratings(self):
        return csr_matrix(self.urm.dot(self.sym))
