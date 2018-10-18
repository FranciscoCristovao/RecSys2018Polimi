import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

class cbfRS:

    icm = pd.DataFrame()
    sym = pd.DataFrame()

    def __init__(self, icm):
        print("CBF recommender has been initialized")
        self.icm = icm
        self.tracks = len(icm)

    def fit(self):
        print("Fitting...")
        # print(self.icm.describe)
        self.sym = pd.DataFrame(cosine_similarity(self.icm, self.icm, dense_output=False))
        # print(self.sym)
        # self.sym.to_csv('data/sym_matrix.csv', sep=',')

    def recommend(self, user_preferences, p_id):
        print("Recommending...")
        pred = {}  # pd.DataFrame([])
        print(p_id)

        for k in p_id:

            # temp is the songs that user liked, Ruj
            ruj = user_preferences.loc[user_preferences['playlist_id'] == k]
            # the similarity of the song


            # todo: new way to find Sij, take ICM select only j rows, then use methods to sum
            df_sij = pd.DataFrame(self.sym.loc[ruj.values[0][1]])

            for j in ruj.values:
                index_j = j[1]
                df_sij[index_j] = self.sym.loc[index_j]
                # sum of all columns
                # r_pred_all_sij.append(df_sij[index_j].sum())
            # sum of all rows
            df_sij['s_ij'] = df_sij.sum(axis=1)
            # print(df_sij)
            # print(r_pred_all_sij)
            # todo: find the proper way to get top 10 elements
            '''for count in range(10):
                # el = df_sij.loc[df_sij['s_ij'].idxmax()].name
                el = df_sij['s_ij'].idxmax()
                temp.append(el)  # this gave only 1
                df_sij.drop([el], inplace=True)
                # print(df_sij[el])
            '''
            df_sij = df_sij.sort_values(['s_ij'], ascending=False).head(10).index

            temp = df_sij.values

            # df.groupby('State')['Population'].apply(lambda grp: grp.nlargest(10).sum())
            '''
            # we select then Sji
            for i in range(self.tracks):
                if i % 500 == 0:
                    print('Song#: ', i, '/20634')

                # i skip the songs in the playlist
                if i in ruj['track_id']:
                    continue

                sum_sji = 0
                for j in ruj.values:
                    # j[0] is the playlist, j[1] is the track j. Sj is already a vector
                    sj = self.sym.loc[j[1]][i]
                    sum_sji = sum_sji + sj.sum()

                r_pred_ui.loc[i] = [int(k), int(i), sum_sji]

                r_pred_i.append(sum_sji)
                r_pred_all_sij.append(sum_sji)
                # r_pred_i.append(i)
                r_pred_i.sort(reverse=True)
                r_pred_i = r_pred_i[0:10]

                # print(r_pred_ui)

                # print(temp)
            
            temp = []
            for el in r_pred_i:
                temp.append(np.where(r_pred_all_sij == el)[0][0])
            '''

            string = ' '.join(str(e) for e in temp)
            pred.update({k: string})
            # print(pred)
            print("Playlist num", k, "/50424")

        # print(pd.DataFrame(list(pred.items()), columns=['playlist_id', 'track_ids']))
        return pd.DataFrame(list(pred.items()), columns=['playlist_id', 'track_ids'])
