from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

evaluator = Evaluator()
rs = CbfRS(tracks_data, 10, 10, 10, tf_idf=True,
           weight_album=3, weight_artist=1, use_duration=False)
rs.fit(train_data)
predictions = rs.recommend(target_data['playlist_id'])
map_ = (evaluator.evaluate(predictions, test_data))

save_dataframe('output/content_w_tuning.csv', ',', predictions)
