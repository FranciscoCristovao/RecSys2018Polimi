import pandas as pd
import numpy as np

class Helper:

    def __init__(self):
        print("TopPop recommender has been initialized")

    def buildURMMatrix(self, data):
        d = {'playlist_id': [1, 1, 2], 'song_id': [21, 32, 64]}
        df = pd.DataFrame(data=d)
        '''df = df.groupby('playlist_id').agg(lambda x: x.tolist())
        urm = df.iloc[:,0].str.replace(' ','').str.get_dummies(sep=',')'''
        df2 = df["song_id"]
        print(df2)
        aux = pd.Series(df2.tolist())
        print(aux)
        df = pd.get_dummies(aux)
