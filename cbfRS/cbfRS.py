import numpy as np
import pandas as pd


class cbfRS:

    train_data = pd.DataFrame()

    def __init__(self, train_data):
        print("CBF recommender has been initialized")
        self.train_data = train_data

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating...")

    def recommend(self, playlist_ids):
        print("Recommending...")