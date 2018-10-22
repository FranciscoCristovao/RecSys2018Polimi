import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper
from utils.cosine_similarity import Cosine
from utils.auxUtils import check_matrix


class CbfRS:

    helper = Helper()
    train_data = pd.DataFrame()

    def __init__(self, data, k=50, shrinkage=100, similarity='cosine'):
        self.k = k
        self.shrinkage = shrinkage
        self.similarity_name = similarity

        if similarity == 'cosine':
            self.cosine = Cosine(shrinkage=self.shrinkage)
        '''
        elif similarity == 'pearson':
            self.distance = Pearson(shrinkage=self.shrinkage)
        elif similarity == 'adj-cosine':
            self.distance = AdjustedCosine(shrinkage=self.shrinkage)
        else:
            raise NotImplementedError('Distance {} not im')
        '''
        print("CBF recommender has been initialized")

        data = data.drop(columns="duration_sec")
        self.icm = self.helper.buildICMMatrix(data)

        # print(self.icm.todense())
        print("ICM loaded into the class")

    def fit(self, train_data):
        print("Fitting...")
        self.train_data = train_data
        # it was a numpy array, i transformed it into a csr matrix
        # Here we have 3 different ways to compute the similarities
        # self.sym = csr_matrix(self.icm.dot(self.icm.T))
        self.sym = self.cosine.compute(self.icm)
        self.sym = check_matrix(self.sym, 'csr')
        # print(type(self.sym))
        # self.sym = self.cos.compute(self.icm, 0)
        # self.sym = csr_matrix(cosine_similarity(self.icm, self.icm))
        # print(self.sym)
        print("Sym correctly loaded")
        self.urm = self.helper.buildURMMatrix(train_data)

    def recommend(self, playlist_ids):
        print("Recommending...")

        final_prediction = {}  # pd.DataFrame([])

        print("STARTING ESTIMATION")
        # add ravel() ?
        estimated_ratings = csr_matrix(self.urm.dot(self.sym)).toarray()
        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]

            # aux contains the indices (track_id) of the most similar songs
            aux = row.argsort()[::-1]
            top_songs = self._filter_seen(k, aux)[:10]
            if len(top_songs) < 10:
                print("Francisco was right once at least")

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})

            if(counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def _filter_seen(self, p_id, ranking):
        seen = self.train_data['track_id'].loc[self.train_data['playlist_id'] == p_id].values
        unseen_mask = np.in1d(ranking, seen, assume_unique=True, invert=True)
        return ranking[unseen_mask]
