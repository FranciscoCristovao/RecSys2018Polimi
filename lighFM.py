from lightfm import LightFM
from lightfm.data import Dataset
from loader.loader import test_data, train_data, target_data, tracks_data
from utils.auxUtils import buildFMMatrix, buildURMMatrix, Evaluator
from scipy.sparse import coo_matrix
import numpy as np
import pandas as pd
from tqdm import tqdm

e = Evaluator()
# Instantiate and train the model
alpha= 1e-3
model = LightFM(no_components=30, loss='warp', learning_rate=0.01)
# todo add latent factors weights
# todo force the dimention of the data matrix
urm = coo_matrix(buildURMMatrix(train_data))
print('Fitting...')
# todo: item features
model.fit(urm, epochs=30, num_threads=4, item_features=item_feature)
final_prediction = {}
tracks = np.array(tracks_data['track_id'], dtype='int32')
for k in tqdm(target_data['playlist_id']):
    # user_index = np.full(len(tracks), k, dtype='int32')
    predictions = model.predict(k, tracks)
    ranking = (np.argsort(predictions)[::-1])[:10]
    string = ' '.join(str(e) for e in ranking)
    final_prediction.update({k: string})


df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
print(df)
print(e.evaluate(df, test_data))

dataset = Dataset(item_identity_features=False)
item_features = dataset.build_item_features(((x['track_id'], [x['album_id']])
                                              for x in tracks_data))

item_features = dataset.build_item_features(((x['track_id'])
                                             for index, x in tracks_data.iterrows()))
