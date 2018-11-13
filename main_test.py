import time

from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator, buildURMMatrix, filter_seen
from hybrid_col_cbf_RS.hybridRS_2 import HybridRS

rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)
predictions = rs.recommend(target_data["playlist_id"])

evaluator.evaluate(predictions, test_data)
save_dataframe('output/hybrid_output.csv', ',', predictions)
