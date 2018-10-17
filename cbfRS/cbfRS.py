import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class cbfRS:

    icm = pd.DataFrame()
    sym = pd.DataFrame()

    def __init__(self, icm):
        print("CBF recommender has been initialized")
        self.icm = icm

    def fit(self):
        print("Fitting...")
        self.sym = pd.DataFrame(cosine_similarity(self.icm, self.icm))

    def evaluate(self):
        print("Evaluating...")

    def recommend(self, playlist_ids):
        print("Recommending...")