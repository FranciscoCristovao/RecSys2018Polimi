import time

from slimRS.slimRS import SLIM_BPR_Recommender
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator, buildURMMatrix, filter_seen
import pandas as pd
# from lightfm import LightFM
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from hybrid_col_cbf_RS.hybridRS import HybridRS
from matrixFactorizationRS.matrix_factorizationRS import MF_BPR_Cython
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from hybrid_col_cbf_RS.hybrid_slim import HybridRS
import matplotlib.pyplot as plt


map_list = []
t_list = []
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
theta = [0.5, 0.8, 1, 1.2, 1.4, 1.6, 2, 5]
for t in theta:
    predictions = rs.recommend(target_data["playlist_id"], theta=t)
    map_ = evaluator.evaluate(predictions, test_data)
    print("THETA: ", t)
    print("map: ", map_)
    map_list.append(map_)
    t_list.append(t)

plt.plot(t_list, map_list)
plt.xlabel('Theta')
plt.ylabel('map')
plt.title('Theta Tuning')
plt.grid(True)
plt.show()

'''
for b in beta:
    print("Beta: ", b)

    predictions = rs.recommend(target_data["playlist_id"], beta=b)
    map_ = evaluator.evaluate(predictions, test_data)
    map_list.append(map_)
    b_list.append(b)

plt.plot(b_list, map_list)
plt.xlabel('Beta (item+slim weight)')
plt.ylabel('map')
plt.title('Beta Tunning')
plt.grid(True)
plt.show()
'''
'''
rs = SLIM_BPR_Cython(train_data)
rs.fit(lambda_i=0.001, lambda_j=0.001)
predictions = rs.recommend(target_data["playlist_id"])
evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)
save_dataframe('output/slim_cython.csv', ',', predictions)
'''
'''
# Hybrid (cbf - colf)

rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)

predictions = rs.recommend(target_data['playlist_id'], 1, 5, 7)
evaluator.evaluate(predictions, test_data)

save_dataframe('output/hybrid_output.csv', ',', predictions)
'''
'''
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)

df = pd.DataFrame([[0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'map'])
alpha = 1
beta = 1
gamma = 1

while beta <= 10:
    gamma = 1
    while gamma <= 10:
        alpha = 1
        beta = 5
        gamma = 7
        hybrid = rs.recommend(target_data['playlist_id'], alpha, beta, gamma)
        print("Alpha: ", alpha, " Beta: ", beta, "Gamma: ", gamma)
        temp_map = evaluator.evaluate(hybrid, test_data)

        df = df.append(pd.DataFrame([[alpha, beta, gamma, temp_map]],
                                    columns=['alpha', 'beta', 'gamma', 'map']))
        top_20 = df.sort_values(by=['map']).tail(20)

        gamma += 1

        print(top_20)
    beta += 1

predictions = rs.recommend(target_data['playlist_id'], alpha, beta, gamma)
temp_map = evaluator.evaluate(predictions, test_data)
'''
'''
rs = HybridRS(tracks_data, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)
predictions = rs.recommend(target_data["playlist_id"])

evaluator.evaluate(predictions, test_data)
save_dataframe('output/hybrid_output.csv', ',', predictions)
'''
'''
rs = CbfRS(tracks_data, 10, 10, 10, tf_idf=True)
rs.fit(train_data)

evaluator = Evaluator()
predictions = rs.recommend(target_data["playlist_id"])
evaluator.evaluate(predictions, test_data)


save_dataframe('output/cbf_output.csv', ',', predictions)
'''
'''
# Hybrid Coll_i_i CBF
evaluator = Evaluator()
i = 0
rs = HybridRS(tracks_data, 10, 10, 300, 0, tf_idf=True)
rs.fit(train_data)
# best alpha=0.5
predictions = rs.recommend(target_data["playlist_id"], 0.5)
evaluator.evaluate(predictions, test_data)
save_dataframe('output/hybrid_col_i_i_cbf.csv', ',', predictions)
'''

'''
k=20
while k < 400:
    print("K: ", k)
    rs = ColBfIIRS(10, k, 0, tf_idf=True)
    rs.fit(train_data)
    evaluator = Evaluator()
    predictions = rs.recommend(target_data["playlist_id"])
    evaluator.evaluate(predictions, test_data)
    k+=20
save_dataframe('output/cb_u_u_output.csv', ',', predictions)
'''

'''
rs = FunkSVD(train_data)
rs.fit()
final_prediction = {}
for k in target_data['playlist_id']:
    print("Recomending for playlist: ", k)
    top_songs = rs.recommend(k, 10, True)
    string = ' '.join(str(e) for e in top_songs)
    final_prediction.update({k: string})
predictions = pd.DataFrame(list(final_prediction.items()), columns=['qplaylist_id', 'track_ids'])
evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)
save_dataframe('output/funksvd_output.csv', ',', predictions)
'''
