import pandas as pd
import numpy as np

class Helper:

    def __init__(self):
        print("Helper has been initialized")

    def buildURMMatrix(self, data):g
        df = pd.get_dummies(data.set_index('playlist_id')['track_id'])
        df.to_csv("data/URM.csv", index=False, sep=',')
