from hybrid_col_cbf_RS.hybrid_slim import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

#Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10)
evaluator = Evaluator()
rs.fit(train_data)
temp_res = []
gamma_list = []
gamma = 50
max_map = 0
# best gamma looks 100
while gamma < 200:
    max_res = rs.recommend(target_data['playlist_id'], 3, 6, 2, gamma)
    temp = evaluator.evaluate(max_res, test_data)
    temp_res.append(temp)
    gamma_list.append(gamma)
    gamma += 5
    if max_map < temp:
        max_map = temp
        max_pred = max_res
    print(gamma)

plt.plot(gamma_list, temp_res, 'bs')
plt.title("Gamma tuning")
plt.show()
print(max_map)

print('No Slim: ')
no_slim = rs.recommend_noslim(target_data['playlist_id'], 3, 6, 2)
evaluator.evaluate(no_slim, test_data)
# print(df)
save_dataframe('output/hybrid.csv', ',', max_pred)
save_dataframe('output/hybrid_no_slim.csv', ',', no_slim)
