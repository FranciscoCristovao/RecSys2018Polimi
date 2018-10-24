import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper, check_matrix, filter_seen
from utils.cosine_similarity_full import Compute_Similarity_Python


class HybridRS:

    helper = Helper()
    train_data = pd.DataFrame()

    def __init__(self, data, at, k=100, shrinkage=0, similarity='cosine'):

        self.k = k
        self.at = at
        self.shrinkage = shrinkage
        self.similarity_name = similarity

    def fit(self, train_data):
        print('Fitting...')


    def recommend(self, playlist_ids):
        return df

