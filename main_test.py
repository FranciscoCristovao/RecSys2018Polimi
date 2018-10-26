import pandas as pd
import numpy as np
from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from collaborative_filtering_user_userRS.colbf_u_uRS import ColBfUURS
import matplotlib.pyplot as plt

# external libraries

# TOPPOP
'''
rs = TopPopRS()
rs.fit(train_data)
predictions = rs.recommend(target_data)
save_dataframe('output/submission_top_pop.csv', ',', predictions)
'''

# CBF
# rs = CbfRS(tracks_data, 5)

# Collaborative Filter User - User
rs = ColBfUURS(10, 200, 0)
rs.fit(train_data)
predictions = rs.recommend(target_data['playlist_id'])
evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)
save_dataframe('output/col_200_5_second.csv', ',', predictions)
'''

list_knn = [100, 150, 175, 190, 195, 205, 210, 215, 220, 225, 240, 260, 300, 350, 450]
# list_knn = [200, 225]
shrinkage_list = [3, 4, 5, 6, 7, 8, 9, 10]
# shrinkage_list = [0, 1]

for shrinkage in shrinkage_list:
    map_list = []
    list_k = []
    for k in list_knn:
        print("Using: ", k, " knn; ", shrinkage, " shrinkage")
        rs = ColBfUURS(10, k, shrinkage)
        rs.fit(train_data)
        predictions = rs.recommend(target_data['playlist_id'])
        map_list.append(evaluator.evaluate(predictions, test_data))
        list_k.append(k)

    plt.plot(list_k, map_list, 'ro')
    plt.title(shrinkage)
    plt.show()
'''
'''
rs = ColBfUURS(10, 200, 0)
rs.fit(train_data)
predictions = rs.recommend(target_data['playlist_id'])
evaluator.evaluate(predictions, test_data)
'''
'''
print("Trying the hybrid stupidly....")
rs_1 = ColBfUURS(10, 200, 0)
rs_2 = CbfRS(tracks_data, 10)
rs_1.fit(train_data)
rs_2.fit(train_data)
k = 0

final_prediction_2_8 = {}
final_prediction_3_7 = {}
final_prediction_4_6 = {}
final_prediction_5_5 = {}
final_prediction_6_4 = {}
final_prediction_7_3 = {}
final_prediction_8_2 = {}
final_prediction_9_1 = {}
final_prediction_10_0 = {}


for k in target_data['playlist_id']:
    predictions_single_collaborative = rs_1.recommend_single(k)
    predictions_single_content = rs_2.recommend_single(k)

    predictions_2_8 = np.concatenate((predictions_single_collaborative[:2], predictions_single_content[:8]))
    predictions_3_7 = np.concatenate((predictions_single_collaborative[:3], predictions_single_content[:7]))
    predictions_4_6 = np.concatenate((predictions_single_collaborative[:4], predictions_single_content[:6]))
    predictions_5_5 = np.concatenate((predictions_single_collaborative[:5], predictions_single_content[:5]))
    predictions_6_4 = np.concatenate((predictions_single_collaborative[:6], predictions_single_content[:4]))
    predictions_7_3 = np.concatenate((predictions_single_collaborative[:7], predictions_single_content[:3]))
    predictions_8_2 = np.concatenate((predictions_single_collaborative[:8], predictions_single_content[:2]))
    predictions_9_1 = np.concatenate((predictions_single_collaborative[:8], predictions_single_content[:2]))
    predictions_10_0 = predictions_single_collaborative[:10]

    string_2_8 = ' '.join(str(e) for e in predictions_2_8)
    string_3_7 = ' '.join(str(e) for e in predictions_3_7)
    string_4_6 = ' '.join(str(e) for e in predictions_4_6)
    string_5_5 = ' '.join(str(e) for e in predictions_5_5)
    string_6_4 = ' '.join(str(e) for e in predictions_6_4)
    string_7_3 = ' '.join(str(e) for e in predictions_7_3)
    string_8_2 = ' '.join(str(e) for e in predictions_8_2)
    string_9_1 = ' '.join(str(e) for e in predictions_9_1)
    string_10_0 = ' '.join(str(e) for e in predictions_10_0)

    final_prediction_2_8.update({k: string_2_8})
    final_prediction_3_7.update({k: string_3_7})
    final_prediction_4_6.update({k: string_4_6})
    final_prediction_5_5.update({k: string_5_5})
    final_prediction_6_4.update({k: string_6_4})
    final_prediction_7_3.update({k: string_7_3})
    final_prediction_8_2.update({k: string_8_2})
    final_prediction_9_1.update({k: string_9_1})
    final_prediction_10_0.update({k: string_10_0})


df_2_8 = pd.DataFrame(list(final_prediction_2_8.items()), columns=['playlist_id', 'track_ids'])
df_3_7 = pd.DataFrame(list(final_prediction_3_7.items()), columns=['playlist_id', 'track_ids'])
df_4_6 = pd.DataFrame(list(final_prediction_4_6.items()), columns=['playlist_id', 'track_ids'])
df_5_5 = pd.DataFrame(list(final_prediction_5_5.items()), columns=['playlist_id', 'track_ids'])
df_6_4 = pd.DataFrame(list(final_prediction_6_4.items()), columns=['playlist_id', 'track_ids'])
df_7_3 = pd.DataFrame(list(final_prediction_7_3.items()), columns=['playlist_id', 'track_ids'])
df_8_2 = pd.DataFrame(list(final_prediction_8_2.items()), columns=['playlist_id', 'track_ids'])
df_9_1 = pd.DataFrame(list(final_prediction_9_1.items()), columns=['playlist_id', 'track_ids'])
df_10_0 = pd.DataFrame(list(final_prediction_10_0.items()), columns=['playlist_id', 'track_ids'])

map_items = []
map_items.append(evaluator.evaluate(df_2_8, test_data))
map_items.append(evaluator.evaluate(df_4_6, test_data))
map_items.append(evaluator.evaluate(df_5_5, test_data))
map_items.append(evaluator.evaluate(df_3_7, test_data))
map_items.append(evaluator.evaluate(df_6_4, test_data))
map_items.append(evaluator.evaluate(df_7_3, test_data))
map_items.append(evaluator.evaluate(df_8_2, test_data))
map_items.append(evaluator.evaluate(df_9_1, test_data))
map_items.append(evaluator.evaluate(df_10_0, test_data))

plt.plot(map_items, [2, 3, 4, 5, 6, 7, 8, 9, 10], 'ro')
plt.show()
'''
'''
save_dataframe('output/hybrid_pred_2_8.csv', ',', df_2_8)
save_dataframe('output/hybrid_pred_3_7.csv', ',', df_3_7)
save_dataframe('output/hybrid_pred_4_6.csv', ',', df_4_6)
save_dataframe('output/hybrid_pred_5_5.csv', ',', df_5_5)
save_dataframe('output/hybrid_pred_7_3.csv', ',', df_7_3)
save_dataframe('output/hybrid_pred_8_2.csv', ',', df_8_2)

save_dataframe('output/hybrid_pred_6_4.csv', ',', df_6_4)
'''
'''
shrink = 0
range_k = [100, 120, 140, 160, 180, 200, 225, 250, 275, 300, 350, 400, 500]
stat = {}
while shrink in range(10):
    map_list = []
    print(shrink)
    for k in range_k:
        rs = ColBfUURS(10, k, shrink)
        rs.fit(train_data)
        predictions = rs.recommend(target_data['playlist_id'])
        print("The k i am using is:", k)
        map_list.append(evaluator.evaluate(predictions, test_data))

    stat.update({shrink: map_list})
    shrink += 2

pd.DataFrame(stat, columns=['shrink', 'map']).to_csv('shrink_k_ratio')
'''
