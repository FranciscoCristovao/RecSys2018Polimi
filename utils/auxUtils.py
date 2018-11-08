import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
import time, sys
import scipy
from sklearn.model_selection import train_test_split


def buildURMMatrix(data):

    playlists = data["playlist_id"].values
    tracks = data["track_id"].values
    interaction = np.ones(len(tracks))
    coo_urm = coo_matrix((interaction, (playlists, tracks)))
    # print("This is the coo_urm", coo_urm)
    return coo_urm.tocsr()


def buildICMMatrix(data):

    '''
    tracks = data["track_id"].values
    artists = data["artist_id"].values
    interaction = np.ones(len(tracks))
    coo_icm = coo_matrix((interaction, (tracks, artists)))


    print("Coo icm with artists correctly built.")  # , features_per_item.shape)
    # print("Item per features: ", items_per_feature.shape)

    return coo_icm.tocsr()
    '''
    '''

    tracks = data["track_id"].values
    albums = data["album_id"].values
    artists = data["artist_id"].values
    features = np.concatenate([albums, artists])
    tracks_sized = np.concatenate([tracks, tracks])
    interaction = np.ones(len(features))
    coo_icm = coo_matrix((interaction, (tracks_sized, features)))
    return coo_icm.tocsr()
    '''

    frames = [pd.get_dummies(data['album_id']), pd.get_dummies(data['artist_id'])]
    icm = pd.concat(frames, axis=1)
    print("ICM with artists and albums correctly built.")

    return csr_matrix(icm)


def dataframeToCSR(data):
    print(csr_matrix(data))


class Cosine:

    def compute(self, mat, shrinkage):
        # convert to csc matrix for faster column-wise operations
        mat = mat.tocsc()
        # print(type(mat))

        # 2) compute the cosine similarity using the dot-product
        dist = mat * mat.T
        print("Computed")

        # zero out diagonal values
        dist = dist - sps.dia_matrix((dist.diagonal()[scipy.newaxis, :], [0]), shape=dist.shape)
        print("Removed diagonal")
        '''
        #SHRINKAGE FOR LATER
        # and apply the shrinkage
        if shrinkage > 0:
            dist = self.apply_shrinkage(X, dist)
            print("Applied shrinkage")
        '''
        return csr_matrix(dist)

    def apply_shrinkage(self, icm, dist):
        # create an "indicator" version of X (i.e. replace values in X with ones)
        icm_ind = icm.copy()
        icm_ind.data = np.ones_like(icm_ind.data)
        # compute the co-rated counts
        co_counts = icm_ind * icm_ind.T
        # remove the diagonal
        co_counts = co_counts - sps.dia_matrix((co_counts.diagonal()[scipy.newaxis, :], [0]), shape=co_counts.shape)
        # compute the shrinkage factor as co_counts_ij / (co_counts_ij + shrinkage)
        # then multiply dist with it
        co_counts_shrink = co_counts.copy()
        co_counts_shrink.data += self.shrinkage
        co_counts.data /= co_counts_shrink.data
        dist.data *= co_counts.data
        return dist


class Evaluator:


    def __init__(self):
        print("Evaluator has been initialized")

    def map(self, recommended_items, relevant_items):

        is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

        '''
        # i have to consider also the position. First i find the elements in the right position
        temp = np.where(recommended_items == relevant_items)
        # i make the same metrics they were using, considering the properly ordered elements
        is_relevant = np.in1d(temp, relevant_items, assume_unique=True)
        print(is_relevant)
        '''

        # Cumulative sum: precision at 1, at 2, at 3 ...
        try:
            p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))
            map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])
        except RuntimeWarning:
            print("Runtime Warning encountered")
            map_score = 0

        return map_score

    def evaluate(self, recommended, test_data):
        cumulative_map = 0.0
        num_eval = 0
        counter = 0
        urm_test = buildURMMatrix(test_data)

        for i in recommended["playlist_id"]:
            try:
                relevant_items = urm_test[i].indices
                # relevant_items = self.get_user_relevant_items(test_user)
            except IndexError:
                print("No row in the test set")
                continue

            if len(relevant_items) > 0:
                recommended_items = np.fromstring(recommended["track_ids"][counter], dtype=int, sep=' ')
                num_eval += 1

                cumulative_map += self.map(recommended_items, relevant_items)

            counter += 1

        cumulative_map /= num_eval
        print("Evaluated", num_eval, "playlists")

        print("Recommender performance is: MAP = {:.4f}".format(cumulative_map))
        return cumulative_map


def check_matrix(X, format, dtype=np.float32):
    if format == 'csc' and not isinstance(X, sps.csc_matrix):
        return X.tocsc().astype(dtype)
    elif format == 'csr' and not isinstance(X, sps.csr_matrix):
        return X.tocsr().astype(dtype)
    elif format == 'coo' and not isinstance(X, sps.coo_matrix):
        return X.tocoo().astype(dtype)
    elif format == 'dok' and not isinstance(X, sps.dok_matrix):
        return X.todok().astype(dtype)
    elif format == 'bsr' and not isinstance(X, sps.bsr_matrix):
        return X.tobsr().astype(dtype)
    elif format == 'dia' and not isinstance(X, sps.dia_matrix):
        return X.todia().astype(dtype)
    elif format == 'lil' and not isinstance(X, sps.lil_matrix):
        return X.tolil().astype(dtype)
    else:
        return X.astype(dtype)


def filter_seen(new_songs, user_playlist):
    seen = user_playlist.indices
    unseen_mask = np.in1d(new_songs, seen, assume_unique=False, invert=True)
    return new_songs[unseen_mask]


def filter_seen_array(new_songs, playlist):
    unseen_mask = np.in1d(new_songs, playlist, assume_unique=True, invert=True)
    return new_songs[unseen_mask]


def filter_seen_array(new_songs, playlist):
    unseen_mask = np.in1d(new_songs, playlist, assume_unique=True, invert=True)
    return new_songs[unseen_mask]


def split_data(full_data, playlists, test_size):
    # take away last song in the playlist...
    ordered_playlist = playlists['playlist_id'][:5000]
    # print("Last element of ordered_playlist: ", ordered_playlist[4999])
    random_playlist = playlists['playlist_id'][5000:]

    train_data = pd.DataFrame(columns=['playlist_id', 'track_id'])
    test_data = pd.DataFrame(columns=['playlist_id', 'track_id'])
    for i in range(50445):
        if i in ordered_playlist.values:
            playlist_ordered = full_data.loc[full_data['playlist_id'] == i]
            chunk_value = int(len(playlist_ordered)*(1-test_size))
            train_single = playlist_ordered.iloc[:chunk_value]
            test_single = playlist_ordered.iloc[chunk_value:]
        else:
            playlist_i = full_data.loc[full_data['playlist_id'] == i]
            train_single, test_single = train_test_split(playlist_i, test_size=test_size)

        if (i % 1000) == 0:
            print("Splitting the dataframe...", i)

        train_data = train_data.append(train_single)
        test_data = test_data.append(test_single)

    return train_data, test_data


def randomization_split(full_dataset, playlists, test_size):
    # take the sequential order and make it random.
    # select only target playlist data
    full_data = pd.merge(full_dataset, playlists, on='playlist_id')

    temp_train, test_data = train_test_split(full_data, test_size=test_size)

    '''train_data = pd.merge(full_dataset, test_data, indicator=True, how='outer') \
        .query('_merge=="left_only"').drop('_merge', axis=1)
    '''

    # take away last song in the playlist...
    ordered_playlist = playlists['playlist_id'][:5000]
    # print("Last element of ordered_playlist: ", ordered_playlist[4999])
    random_playlist = playlists['playlist_id'][5000:]
    temp_list = []
    # print(ordered_playlist)
    for k in playlists['playlist_id']:
        if k in ordered_playlist:
            try:
                playlist_to_order_len = len(test_data['track_id'].loc[test_data['playlist_id'] == k])
                proper_order = full_data['track_id'].loc[full_data['playlist_id'] == k]
                songs_to_draw = proper_order[-playlist_to_order_len:][::-1]
                for i in songs_to_draw:
                    temp_list.append([k, i])
            except:
                print('No such playlist, or other bad things happened while splitting the data')
        else:  # we are in not ordered playlist, so we just add it to the dictionary
            try:
                songs_to_draw = test_data['track_id'].loc[test_data['playlist_id'] == k]
                for i in songs_to_draw:
                    temp_list.append([k, i])
            except:
                print("No such playlist, splitting the data")

    # print(temp_dictionary)
    test_data = pd.DataFrame(temp_list, columns=['playlist_id', 'track_id'])

    print(test_data.head(5))
    '''
    ordered_playlist = playlists['playlist_id'][:5000]
    # print("Last element of ordered_playlist: ", ordered_playlist[4999])
    random_playlist = playlists['playlist_id'][5000:]
    temp_list = []
    # print(ordered_playlist)
    for k in playlists['playlist_id']:
        if k in ordered_playlist:
            try:
                playlist_to_order = test_data['track_id'].loc[test_data['playlist_id'] == k]
                proper_order = full_data['track_id'].loc[full_data['playlist_id'] == k]

                mask = np.in1d(proper_order, playlist_to_order)
                proper_order_to_df = proper_order[mask]
                for i in proper_order_to_df:
                    temp_list.append([k, i])
            except:
                print('No such playlist, or other bad things happened while splitting the data')
        else:  # we are in not ordered playlist, so we just add it to the dictionary
            try:
                playlist_to_order = test_data['track_id'].loc[test_data['playlist_id'] == k]
                for i in playlist_to_order:
                    temp_list.append([k, i])
            except:
                print("No such playlist, splitting the data")

    # print(temp_dictionary)
    test_data_proper = pd.DataFrame(temp_list, columns=['playlist_id', 'track_id'])
    # print(test_data.head(5))
    # print(test_data_proper.head(5))
    print("Data correctly splitted")
    '''

    train_data = pd.concat([full_dataset, test_data, test_data]).drop_duplicates(keep=False)

    return train_data, test_data


def similarityMatrixTopK(item_weights, forceSparseOutput = True, k=100, verbose = False, inplace=True):
    """
    The function selects the TopK most similar elements, column-wise

    :param item_weights:
    :param forceSparseOutput:
    :param k:
    :param verbose:
    :param inplace: Default True, WARNING matrix will be modified
    :return:
    """

    assert (item_weights.shape[0] == item_weights.shape[1]), "selectTopK: ItemWeights is not a square matrix"

    start_time = time.time()

    if verbose:
        print("Generating topK matrix")

    nitems = item_weights.shape[1]
    k = min(k, nitems)

    # for each column, keep only the top-k scored items
    sparse_weights = not isinstance(item_weights, np.ndarray)

    if not sparse_weights:

        idx_sorted = np.argsort(item_weights, axis=0)  # sort data inside each column

        if inplace:
            W = item_weights
        else:
            W = item_weights.copy()

        # index of the items that don't belong to the top-k similar items of each column
        not_top_k = idx_sorted[:-k, :]
        # use numpy fancy indexing to zero-out the values in sim without using a for loop
        W[not_top_k, np.arange(nitems)] = 0.0

        if forceSparseOutput:
            W_sparse = sps.csr_matrix(W, shape=(nitems, nitems))

            if verbose:
                print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

            return W_sparse

        if verbose:
            print("Dense TopK matrix generated in {:.2f} seconds".format(time.time()-start_time))

        return W

    else:
        # iterate over each column and keep only the top-k similar items
        data, rows_indices, cols_indptr = [], [], []

        item_weights = check_matrix(item_weights, format='csc', dtype=np.float32)

        for item_idx in range(nitems):

            cols_indptr.append(len(data))

            start_position = item_weights.indptr[item_idx]
            end_position = item_weights.indptr[item_idx+1]

            column_data = item_weights.data[start_position:end_position]
            column_row_index = item_weights.indices[start_position:end_position]

            idx_sorted = np.argsort(column_data)  # sort by column
            top_k_idx = idx_sorted[-k:]

            data.extend(column_data[top_k_idx])
            rows_indices.extend(column_row_index[top_k_idx])


        cols_indptr.append(len(data))

        # During testing CSR is faster
        W_sparse = sps.csc_matrix((data, rows_indices, cols_indptr), shape=(nitems, nitems), dtype=np.float32)
        W_sparse = W_sparse.tocsr()

        if verbose:
            print("Sparse TopK matrix generated in {:.2f} seconds".format(time.time() - start_time))

        return W_sparse


def buildICMMatrix_comeback(data):

    '''
    tracks = data["track_id"].values
    artists = data["artist_id"].values
    interaction = np.ones(len(tracks))
    coo_icm = coo_matrix((interaction, (tracks, artists)))


    print("Coo icm with artists correctly built.")  # , features_per_item.shape)
    # print("Item per features: ", items_per_feature.shape)

    return coo_icm.tocsr()
    '''
    '''

    tracks = data["track_id"].values
    albums = data["album_id"].values
    artists = data["artist_id"].values
    features = np.concatenate([albums, artists])
    tracks_sized = np.concatenate([tracks, tracks])
    interaction = np.ones(len(features))
    coo_icm = coo_matrix((interaction, (tracks_sized, features)))
    return coo_icm.tocsr()
    '''
    # frames = [pd.get_dummies(data['album_id']), pd.get_dummies(data['artist_id']), pd.get_dummies(data['duration_sec'])]
    frames = [pd.get_dummies(data['album_id'])]
    icm = pd.concat(frames, axis=1)

    print("ICM with artists and albums correctly built.")

    return csr_matrix(icm)

