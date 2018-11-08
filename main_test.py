from hybrid_col_cbf_RS.hybrid_slim import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator

#Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10)
evaluator = Evaluator()
rs.fit(train_data)
max_res = rs.recommend(target_data['playlist_id'], 3, 6, 2, 20)
temp = evaluator.evaluate(max_res, test_data)
print('No Slim: ')
no_slim = rs.recommend_noslim(target_data['playlist_id'], 3, 6, 2)
evaluator.evaluate(no_slim, test_data)
# print(df)
save_dataframe('output/hybrid.csv', ',', max_res)