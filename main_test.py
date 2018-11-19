from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from hybrid_col_cbf_RS.hybridRS import HybridRS

# Hybrid (cbf - colf)
# provo sgd, rms prop per vedere qual è il migliore
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(full_data)

alpha = 1
beta = 5
gamma = 7

predictions = rs.recommend(target_data['playlist_id'], alpha, beta, gamma, delta=0.9)
evaluator.evaluate(predictions, test_data)

save_dataframe('output/hybrid_adam.csv', ',', predictions)
