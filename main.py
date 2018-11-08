import time

from slimRS.slimRS import SLIM_BPR_Recommender
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator, buildURMMatrix, filter_seen
import pandas as pd
from lightfm import LightFM


# SLIM-BPR

rs = SLIM_BPR_Recommender(train_data)
rs.fit()

final_prediction = {}

for k in target_data['playlist_id']:
    print("Recomending for playlist: ", k)
    top_songs = rs.recommend(k, 10, True)
    string = ' '.join(str(e) for e in top_songs)
    final_prediction.update({k: string})

predictions = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
evaluator = Evaluator()

evaluator.evaluate(predictions, test_data)
save_dataframe('output/slim_output.csv', ',', predictions)

'''
rs = FunkSVD(train_data)
rs.fit()

final_prediction = {}

for k in target_data['playlist_id']:
    print("Recomending for playlist: ", k)
    top_songs = rs.recommend(k, 10, True)
    string = ' '.join(str(e) for e in top_songs)
    final_prediction.update({k: string})

predictions = pd.DataFrame(list(final_prediction.items()), columns=['qplaylist_id', 'track_ids'])
evaluator = Evaluator()

evaluator.evaluate(predictions, test_data)
save_dataframe('output/funksvd_output.csv', ',', predictions)
'''
'''
urm = buildURMMatrix(full_data)
urm_train = buildURMMatrix(train_data)

model = LightFM(loss='warp')
model.fit(urm_train, epochs=30)

final_prediction = {}
print(urm.shape)
print(urm_train.shape)

counter = 0
tracks = urm_train.indices

for k in target_data['playlist_id']:
    print("Playlist: ", k, " Counter: ", counter)
    row = model.predict(k, tracks)
    aux = row.argsort()[::-1]
    user_playlist = urm[k]

    top_songs = filter_seen(aux, user_playlist)[:10]
    print(top_songs)

    string = ' '.join(str(e) for e in top_songs)
    final_prediction.update({k: string})

    if (counter % 100) == 0:
        print("Playlist num", counter, "/10000")

    counter += 1

predictions = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
evaluator = Evaluator()

evaluator.evaluate(predictions, test_data)
save_dataframe('output/funksvd_output.csv', ',', predictions)
'''
