import pandas as pd
import numpy as np
from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
import matplotlib.pyplot as plt
# from slimRS.slimRS import SLIM_BPR

# SLIM
'''
rs = SLIM_BPR(train_data)
rs.fit()
prediction = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()
evaluator.evaluate(prediction, test_data)

save_dataframe('output/slim_bpr_.csv', ',', prediction)
'''
# CBF
# rs = CbfRS(tracks_data, 5)

# Collaborative Filter User - User
# list_knn = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
# 500 / 700, 200?
# list_knn = [50, 100, 125, 150, 175, 200, 225, 250, 300]
# list_knn = [200]
# shrinkage_list = [0]
# shrinkage_list = [10, 20, 30, 40, 50, 60, 70, 80, 100][::-1]

evaluator = Evaluator()
df = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map_u_u', 'shr'])
top_10 = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map_u_u', 'shr'])
# print(df)
shrinkage = 3
while shrinkage < 150:
    map_list_u_u = []
    map_list_i_i = []
    list_k = []
    list_df = []
    k = 40
    while k < 600:
        print("Using: ", k, " knn; ", shrinkage, " shrinkage")
        rs = ColBfUURS(10, k, shrinkage)
        # rs_2 = ColBfIIRS(10, k, shrinkage)
        rs.fit(train_data)
        # rs_2.fit(train_data)
        predictions = rs.recommend(target_data['playlist_id'])
        # predictions_2 = rs_2.recommend(target_data['playlist_id'])
        map_u_u = (evaluator.evaluate(predictions, test_data))
        # map_i_i = evaluator.evaluate(predictions_2, test_data)
        map_list_u_u.append(map_u_u)
        # map_list_i_i.append(map_i_i)
        list_k.append(k)
        # df = df.append(pd.DataFrame([[k, map_u_u, map_i_i, shrinkage]], columns=['knn', 'map_u_u', 'map_i_i', 'shr']))
        # df = df.append(pd.DataFrame([[k, map_i_i, shrinkage]], columns=['knn', 'map_i_i', 'shr']))
        df = df.append(pd.DataFrame([[k, map_u_u, shrinkage]], columns=['knn', 'map_u_u', 'shr']))
        if k % 200 == 0:
            top_10 = df.sort_values(by=['map_u_u']).tail(10)
            print(top_10)
        k += 10
    print(top_10)
    # plt.plot(list_k, map_list_u_u, 'ro', list_k, map_list_i_i, 'bs')
    plt.plot(list_k, map_list_u_u, 'bs')
    plt.title(shrinkage)
    plt.show()
    shrinkage += 3

save_dataframe('output/map_par_tuning_wide_shrinkage.csv', ',', df)
'''
save_dataframe('output/collaborative_user_user.csv', ',', predictions)
save_dataframe('output/collaborative_item_item.csv', ',', predictions_2)
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





