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
        # print(self.icm.todense())
        print("ICM loaded into the class")


    def fit(self, trainData):
        print("Fitting...")
        # it was a numpy array, i transformed it into a csr matrix
        self.sym = csr_matrix(cosine_similarity(self.icm, self.icm))
        print("Sym correctly loaded")

        self.urm = self.helper.buildURMMatrix(trainData)


    def recommend(self, p_id):
        print("Recommending...")

        pred = {}  # pd.DataFrame([])

        print("STARTING ESTIMATION")
        estimated_ratings = csr_matrix(self.urm.dot(self.sym))
        # todo: eliminate duplicates
        counter = 0
        #todo: eliminate duplicates
        for k in p_id:

            '''print("Ratings for the first row, reversed:",
                  estimated_ratings.data[estimated_ratings.indptr[k]:estimated_ratings.indptr[k+1]])'''
            row = estimated_ratings[counter]
            # print("Row.data", row.data)
            # aux contains the indexes to sort an array
            aux = np.sort(np.argsort(row.data))
            # print("Row.data sorted", aux)
            top_10_indexes = aux[-10:]

            string = ' '.join(str(row.indices[e]) for e in top_10_indexes)
            pred.update({k: string})
            print("Playlist num", counter, "/10000")
            counter += 1

            # print(pred)

        df = pd.DataFrame(list(pred.items()), columns = ['playlist_id', 'track_ids'])
        print(df)
        return df
        '''
        for row_index in p_id:
            print(row_index)
            row = estimated_ratings[row_index]
            print("FIRST ROW")
            print(row.todense())
            aux = np.argsort(row.todense()) #gets the indexes of the top products
            print('###', aux)
            
            top_items = aux[-10:]
            print('The top', top_items)
            return
            #Add to recommendation
            string = ' '.join(str(e) for e in top_items)
            pred.update({row_index: string})
            print("Playlist num", counter, "/10000")
            counter += 1
        '''