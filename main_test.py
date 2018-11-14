from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from hybrid_col_cbf_RS.hybrid_slim import HybridRS

# Hybrid (cbf - colf)

rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(full_data)

alpha = 1
beta = 5
gamma = 7

# train 4, a = 1, b = 5, c = 7, d = 0.8, map = 0.0778/0.0771
# train 4, a = 1, b = 5, c = 7, d = 0.9, map = 0.0782/0.0780 (knn200, 800, 500)
# train 4, a = 1, b = 5, c = 7, d = 0.9, map = 0.0783 knn_slim = 600

predictions = rs.recommend(target_data['playlist_id'], alpha, beta, gamma, delta=0.9)
evaluator.evaluate(predictions, test_data)
'''
predictions = rs.recommend_reverse(target_data['playlist_id'], alpha, beta, gamma, delta=0.9)
evaluator.evaluate(predictions, test_data)

predictions = rs.recommend_probability(target_data['playlist_id'], alpha, beta, gamma, p_treshold=0.9)
evaluator.evaluate(predictions, test_data)
'''

save_dataframe('output/hybrid_output_slim_09.csv', ',', predictions)
