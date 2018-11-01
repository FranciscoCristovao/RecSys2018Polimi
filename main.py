from slimRS.slimRS import SLIM_BPR_Recommender
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd

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