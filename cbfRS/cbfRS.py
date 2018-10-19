import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from utils.auxUtils import Helper

class cbfRS:

    icm = pd.DataFrame()
    sym = pd.DataFrame()
    urm = pd.DataFrame()
    helper = Helper()

    def __init__(self, data):
        print("CBF recommender has been initialized")

        data = data.drop(columns="duration_sec")
        self.icm = self.helper.buildICMMatrix(data)
        print(self.icm.todense())


    def fit(self, trainData):
        print("Fitting...")

        self.sym = cosine_similarity(self.icm, self.icm)
        print(self.sym)
        self.urm = self.helper.buildURMMatrix(trainData)


    def recommend(self, p_id):
        print("Recommending...")

        pred = {}  # pd.DataFrame([])


        print("STARTING ESTIMATION")
        estimated_ratings = csr_matrix(self.urm.dot(self.sym))

        print("Estimated ratings!!")
        #ratings_df = pd.DataFrame(estimated_ratings.toarray())

        counter = 1

        for row_index in p_id:

            row = estimated_ratings.getrow(row_index)
            print("FIRST ROW")
            print("Index")
            print(row_index)

            top_items = np.argsort(-row)[:10] #gets the indexes of the top products
            print("SONGS")
            print(top_items)

            return
            #print("Suggested items")
            #print (top_items)
            #if(counter==4): return


            #Add to recommendation
            string = ' '.join(str(e) for e in top_items)
            pred.update({row_index: string})
            print("Playlist num", counter, "/10000")
            counter += 1

        return pd.DataFrame(list(pred.items()), columns=['playlist_id', 'track_ids'])
