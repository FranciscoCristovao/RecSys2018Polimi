import time

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
# from hybrid_col_cbf_RS.hybrid_pureSVD import HybridRS
from svdRS.pureSVD import PureSVDRecommender
import matplotlib.pyplot as plt
from slimRS.slimElasticNet import SLIMElasticNetRecommender

map_list = []
k_list = []
ks = [200, 300, 400, 500, 600, 700]
evaluator = Evaluator()
rs = SLIMElasticNetRecommender(train_data)

for k in ks:
    rs.fit(topK=k)
    predictions = rs.recommend(target_data['playlist_id'])
    map_ = evaluator.evaluate(predictions, test_data)
    print("K: ", k)
    print("map: ", map_)
    map_list.append(map_)
    k_list.append(k)

plt.plot(k_list, map_list)
plt.xlabel('K')
plt.ylabel('map')
plt.title('K ElasticNet Tuning')
plt.grid(True)
plt.show()

'''
map_list = []
d_list = []
deltas = [0, 5, 10, 15, 20]
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
for d in deltas:
    predictions = rs.recommend(target_data['playlist_id'], delta=d)
    map_ = evaluator.evaluate(predictions, test_data)
    print("Delta: ", d)
    print("map: ", map_)
    map_list.append(map_)
    d_list.append(d)

plt.plot(d_list, map_list)
plt.xlabel('Delta')
plt.ylabel('map')
plt.title('Delta SLIM hybrid Tuning')
plt.grid(True)
plt.show()
'''
'''
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
'''
map_list = []
f_list = []

evaluator = Evaluator()
rs = PureSVDRecommender(train_data)
factor = 450
while factor <= 550:
    rs.fit(num_factors=factor)
    predictions = rs.recommend(target_data['playlist_id'])
    map_ = evaluator.evaluate(predictions, test_data)
    map_list.append(map_)
    f_list.append(factor)
    factor += 20

plt.plot(f_list, map_list)
plt.xlabel('N Factors')
plt.ylabel('map')
plt.title('N Factors Tuning')
plt.grid(True)
plt.show()
'''
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
