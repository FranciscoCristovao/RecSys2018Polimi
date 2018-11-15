from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from hybrid_col_cbf_RS.hybridRS import HybridRS

# Hybrid Standard

rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(full_data)

predictions = rs.recommend(target_data['playlist_id'], alpha=1, beta=5, gamma=7, filter_top_pop=True)
evaluator.evaluate(predictions, test_data)

save_dataframe('output/hybrid_no_slim_filtering_top_pop.csv', ',', predictions)
