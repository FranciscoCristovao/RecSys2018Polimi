import pandas as pd
import os
import numpy as np


trainData = pd.read_csv('data/train.csv', sep=',')
targetData = pd.read_csv('data/target_playlists.csv', sep=',')


def save_dataframe(path, sep, dataframe):

    dataframe.to_csv(path, index=False, sep=sep)
    print("Successfully built csv..")


'''

def save_topPop_result(path, sep, data):

    csv = open(path, "w")

    columnTitleRow = "playlist_id,track_ids\n"
    csv.write(columnTitleRow)

    for val in targetData['playlist_id'].unique():

        row = np.array2string(val) + "," + np.array2string(data) + "\n"
        csv.write(row) 
        
def save_dataframe_arr(path, sep, arr):
    for i in range(len(arr)):
        arr[i] = {'track_ids': ' '.join(str(e) for e in arr[i])}

    dataframe = pd.DataFrame(arr)
    dataframe.index.names=['playlist_id']

    dataframe.to_csv(path, sep=sep)
    print("Successfully built csv from arr.. ")
       
        



'''