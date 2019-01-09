import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from utils.auxUtils import split_data_fast
from loader.loader import target_data, full_data_sequential
import pandas as pd
import xgboost as xgb
from tqdm import tqdm

train = pd.read_csv('xgboost_data/xgboost_train.csv', ',')
test = pd.read_csv('xgboost_data/xgboost_test.csv', ',')
df = pd.read_csv('xgboost_data/xgboost_start_80.csv', ',')
'''
# user this code to preprocess predictions
list = []
for i in df['playlist_id']:
    tracks = df['track_ids'].loc[df['playlist_id'] == i]
    array = tracks[i].split()
    for j in array:
        list.append({'playlist_id': i, 'track_id': j})
df2 = (pd.DataFrame(list, columns=['playlist_id', 'track_id'])).set_index('playlist_id')
'''

# train['track_id'] = train['track_ids'].apply(lambda x: x.split())
train_data, test_data = split_data_fast(df, full_data_sequential, target_data, test_size=0.2)
train_data = train_data[['playlist_id', 'track_id']]
test_data = test_data[['playlist_id', 'track_id']]

X_train = train_data

y_train = []
all_playlist = np.unique(train_data['playlist_id'])
# todo: make this faster
for i in tqdm(all_playlist):
    tracks_test = test['track_id'].loc[test['playlist_id'] == i]
    tracks_splitted = train_data['track_id'].loc[train_data['playlist_id'] == i]
    mask = np.in1d(tracks_splitted, tracks_test, assume_unique=True, invert=False)
    y_train = np.append(y_train, mask.astype(int)) 

#y_train = train_data['playlist_id'].astype('int32')

# X_test = test_data['track_id'].astype('int32')
X_test = test_data


y_test = []
all_playlist = np.unique(test_data['playlist_id'])
# todo: make this faster
for i in tqdm(all_playlist):
    tracks_test = test['track_id'].loc[test['playlist_id'] == i]
    tracks_splitted = test_data['track_id'].loc[test_data['playlist_id'] == i]
    mask = np.in1d(tracks_splitted, tracks_test, assume_unique=True, invert=False)
    y_test = np.append(y_test, mask.astype(int)) 

# y_test = test_data['playlist_id'].astype('int32')

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'max_depth': 3,  # the maximum depth of each tree
    'eta': 0.3,  # step for each iteration
    'silent': 1,  # keep it quiet
    'objective': 'multi:softprob',  # error evaluation for multiclass training
    'num_class': 3,  # the number of classes
    'eval_metric': 'merror'}  # evaluation metric

num_round = 20  # the number of training iterations (number of trees)


model = xgb.train(params,
                  dtrain,
                  num_round,
                  verbose_eval=2,
                  evals=[(dtrain, 'train')])

from sklearn.metrics import precision_score

preds = model.predict(dtest)
best_preds = np.asarray([np.argmax(line) for line in preds])
print("Precision: {:.2f} %".format(precision_score(y_test, best_preds, average='macro')))

