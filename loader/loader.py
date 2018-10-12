import pandas as pd


def load_dataframe(path, sep):
    df = pd.read_csv(path, sep=sep)
    return df


def save_dataframe_arr(path, sep, arr):
    for i in range(len(arr)):
        arr[i] = {'track_ids': ' '.join(str(e) for e in arr[i])}

    dataframe = pd.DataFrame(arr)
    dataframe.index.names=['playlist_id']


    #DataFrame()
    print(dataframe)
    dataframe.to_csv(path, sep=sep)
    print("Successfully built csv..")

