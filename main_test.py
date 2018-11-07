from hybrid_col_cbf_RS.hybrid_p_sampling import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator

#Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10)
evaluator = Evaluator()
rs.fit(train_data)
max_res = rs.recommend(target_data['playlist_id'], 3, 6, 2)
temp = evaluator.evaluate(max_res, test_data)

# print(df)
save_dataframe('output/hybrid_i_i_u_u_3_6_2.csv', ',', max_res)
