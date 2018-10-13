import numpy as np
from loader.loader import trainData


class TopPopRS:

    topSongs = np.arange(0, 9)

    def __init__(self):
        print("TopPop recommender has been initialized")

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating..")

    def recommend(self):

        topSongs = trainData['track_id'].value_counts().head(10).index.values

        return topSongs

    def recommend_prop(self, train_data):

        topSongs = train_data['track_id'].value_counts().head(10).index.values

        return topSongs
