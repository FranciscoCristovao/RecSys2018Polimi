import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sps
from sklearn.linear_model import ElasticNet
from tqdm import tqdm
import time, sys

def buildURMMatrix(data):

    playlists = data["playlist_id"].values
    tracks = data["track_id"].values
    interaction = np.ones(len(tracks))
    coo_urm = coo_matrix((interaction, (playlists, tracks)))

    return coo_urm.tocsr()
def split_data_fast(full_data, sequential_data, target_data, test_size):

    # items_threshold is the minimum number of tracks to be in a playilist
    data = full_data

    sequential_index = target_data[:5000]['playlist_id'].values
    sequential_mask = data['playlist_id'].isin(sequential_index)  # .reset_index()['playlist_id']

    random_data = data[~sequential_mask]

    # not so good, list / len does not work

    train_data_sequential = sequential_data.groupby('playlist_id', as_index=False).\
        apply(lambda x: x[:int(len(x)*(1 - test_size))]).reset_index(drop=True)

    test_data_sequential = sequential_data.groupby('playlist_id', as_index=False).\
        apply(lambda x: x[int(len(x)*(1 - test_size)):]).reset_index(drop=True)

    random_data_shuffled = random_data.sample(frac=1)
    # random_data.groupby('playlist_id').apply(lambda x: x['track_id'].sample(len(x)))

    train_data_shuffled = random_data_shuffled.groupby('playlist_id', as_index=False).\
        apply(lambda x: x[:(int(len(x)*(1 - test_size)))]).reset_index(drop=True)
    # with list it's faster
    test_data_shuffled = random_data_shuffled.groupby('playlist_id', as_index=False).\
        apply(lambda x: x[(int(len(x)*(1 - test_size))):]).reset_index(drop=True)

    test_data = test_data_shuffled.append(test_data_sequential)
    train_data = train_data_shuffled.append(train_data_sequential)

    return train_data, test_data
def save_dataframe(path, sep, dataframe):
    dataframe.to_csv(path, index=False, sep=sep)
    print("Successfully built csv..")
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
class SLIMElasticNetRecommender():
    """
    Train a Sparse Linear Methods (SLIM) item similarity model.
    NOTE: ElasticNet solver is parallel, a single intance of SLIM_ElasticNet will
          make use of half the cores available
    See:
        Efficient Top-N Recommendation by Linear Regression,
        M. Levy and K. Jack, LSRS workshop at RecSys 2013.
        https://www.slideshare.net/MarkLevy/efficient-slides
        SLIM: Sparse linear methods for top-n recommender systems,
        X. Ning and G. Karypis, ICDM 2011.
        http://glaros.dtc.umn.edu/gkhome/fetch/papers/SLIM2011icdm.pdf
    """

    def __init__(self, train_data):

        self.URM_train = buildURMMatrix(train_data)
        self.top_pop_songs = train_data['track_id'].value_counts().head(20).index.values

    def __str__(self):
        return "SLIM (l1_penalty={},l2_penalty={},positive_only={})".format(self.l1_penalty, self.l2_penalty, self.positive_only)

    def fit(self, l1_ratio=0.00001, positive_only=True, topK=100):

        self.positive_only = positive_only
        self.topK = topK
        '''
        if self.l1_penalty + self.l2_penalty != 0:
            self.l1_ratio = self.l1_penalty / (self.l1_penalty + self.l2_penalty)
        else:
            print("SLIM_ElasticNet: l1_penalty+l2_penalty cannot be equal to zero, setting the ratio l1/(l1+l2) to 1.0")
            self.l1_ratio = 1.0
        '''
        self.l1_ratio = l1_ratio
        # initialize the ElasticNet model
        self.model = ElasticNet(alpha=1.0,
                                l1_ratio=self.l1_ratio,
                                positive=self.positive_only,
                                fit_intercept=False,
                                copy_X=False,
                                precompute=True,
                                selection='random',
                                max_iter=100,
                                tol=1e-4)

        URM_train = sps.csc_matrix(self.URM_train)

        n_items = URM_train.shape[1]

        # Use array as it reduces memory requirements compared to lists
        dataBlock = 10000000

        rows = np.zeros(dataBlock, dtype=np.int32)
        cols = np.zeros(dataBlock, dtype=np.int32)
        values = np.zeros(dataBlock, dtype=np.float32)

        numCells = 0

        start_time = time.time()
        start_time_printBatch = start_time

        # fit each item's factors sequentially (not in parallel)
        for currentItem in tqdm(range(n_items)):

            # get the target column
            y = URM_train[:, currentItem].toarray()
            # set the j-th column of X to zero
            start_pos = URM_train.indptr[currentItem]
            end_pos = URM_train.indptr[currentItem + 1]

            current_item_data_backup = URM_train.data[start_pos: end_pos].copy()
            URM_train.data[start_pos: end_pos] = 0.0

            # fit one ElasticNet model per column
            self.model.fit(URM_train, y)

            # self.model.coef_ contains the coefficient of the ElasticNet model
            # let's keep only the non-zero values

            # Select topK values
            # Sorting is done in three steps. Faster then plain np.argsort for higher number of items
            # - Partition the data to extract the set of relevant items
            # - Sort only the relevant items
            # - Get the original item index

            # nonzero_model_coef_index = self.model.coef_.nonzero()[0]
            # nonzero_model_coef_value = self.model.coef_[nonzero_model_coef_index]

            nonzero_model_coef_index = self.model.sparse_coef_.indices
            nonzero_model_coef_value = self.model.sparse_coef_.data
            # print(nonzero_model_coef_index)

            local_topK = min(len(nonzero_model_coef_value)-1, self.topK)

            relevant_items_partition = (-nonzero_model_coef_value).argpartition(local_topK)[0:local_topK]
            relevant_items_partition_sorting = np.argsort(-nonzero_model_coef_value[relevant_items_partition])
            ranking = relevant_items_partition[relevant_items_partition_sorting]

            for index in range(len(ranking)):

                if numCells == len(rows):
                    rows = np.concatenate((rows, np.zeros(dataBlock, dtype=np.int32)))
                    cols = np.concatenate((cols, np.zeros(dataBlock, dtype=np.int32)))
                    values = np.concatenate((values, np.zeros(dataBlock, dtype=np.float32)))


                rows[numCells] = nonzero_model_coef_index[ranking[index]]
                cols[numCells] = currentItem
                values[numCells] = nonzero_model_coef_value[ranking[index]]

                numCells += 1

            # finally, replace the original values of the j-th column
            URM_train.data[start_pos:end_pos] = current_item_data_backup

            if time.time() - start_time_printBatch > 300 or currentItem == n_items-1:
                print("Processed {} ( {:.2f}% ) in {:.2f} minutes. Items per second: {:.0f}".format(
                                  currentItem+1,
                                  100.0 * float(currentItem+1)/n_items,
                                  (time.time()-start_time)/60,
                                  float(currentItem)/(time.time()-start_time)))
                sys.stdout.flush()
                sys.stderr.flush()

                start_time_printBatch = time.time()

        # generate the sparse weight matrix
        self.W_sparse = sps.csr_matrix((values[:numCells], (rows[:numCells], cols[:numCells])),
                                       shape=(n_items, n_items), dtype=np.float32)

    def recommend(self, playlist_ids):

        print("Recommending...")

        final_prediction = {}

        estimated_ratings = check_matrix(self.URM_train.dot(self.W_sparse), 'csr')

        counter = 0

        for k in playlist_ids:

            row = estimated_ratings[k]
            if (k == 7):
                print(row.data.sort())
            # aux contains the indices (track_id) of the most similar songs
            indx = row.data.argsort()[::-1]
            aux = row.indices[indx]
            user_playlist = self.URM_train[k]

            aux = np.concatenate((aux, self.top_pop_songs), axis=None)
            top_songs = filter_seen(aux, user_playlist)[:10]

            string = ' '.join(str(e) for e in top_songs)
            final_prediction.update({k: string})
            if (counter % 1000) == 0:
                print("Playlist num", counter, "/10000")

            counter += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        # print(df)
        return df

    def get_estimated_ratings(self):
        return check_matrix(self.URM_train.dot(self.W_sparse), 'csr')



full_data_sequential = pd.read_csv('../input/trainsequentialdata/train_sequential.csv', sep=',')
#  ../input/recommender-system-2018-challenge-polimi/sample_submission.csv

full_data = pd.read_csv('../input/recommender-system-2018-challenge-polimi/train.csv', sep=',')
target_data = pd.read_csv('../input/recommender-system-2018-challenge-polimi/target_playlists.csv', sep=',')
tracks_data = pd.read_csv('../input/recommender-system-2018-challenge-polimi/tracks.csv', sep=',')
train_data, test_data = split_data_fast(full_data, full_data_sequential, target_data, test_size=0.2)

print('')

evaluator = Evaluator()
rs = SLIMElasticNetRecommender(train_data)
l1_ratio = 0.000008
l1_list = []
while l1_ratio < 0.00006:
    rs.fit()
    predictions = rs.recommend(target_data['playlist_id'])
    print(l1_ratio)
    map_temp = evaluator.evaluate(predictions, test_data)
    l1_list.append({'map': map_temp, 'l1_ratio':l1_ratio})
    l1_ratio += 0.000001

print(l1_list, 'knn = 100')
save_dataframe('output/slim_elasticnet.csv', ',', predictions)







