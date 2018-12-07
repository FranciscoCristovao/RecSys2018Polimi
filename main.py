import time

from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator, buildURMMatrix, filter_seen
import pandas as pd
# from lightfm import LightFM
from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from hybrid_col_cbf_RS.hybridRS import HybridRS
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
# from hybrid_col_cbf_RS.hybrid_pureSVD import HybridRS
from svdRS.pureSVD import PureSVDRecommender
import matplotlib.pyplot as plt
from slimRS.slimElasticNet import SLIMElasticNetRecommender
from hybrid_col_cbf_RS.hybrid_knn_slimBPR_elasticNet import HybridRS
from hybrid_col_cbf_RS.hybrid_graph import HybridRS
from hybrid_col_cbf_RS.hybrid_pureSVD import HybridRS

'''
map_list = []
knn_list = []
ks = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 250, 300]
evaluator = Evaluator()
for k in ks:

    rs = CbfRS(tracks_data, k=k)
    rs.fit(train_data)
    predictions = rs.recommend(target_data['playlist_id'])
    map_ = evaluator.evaluate(predictions, test_data)
    print("K: ", k)
    print("MAP: ", map_)
    map_list.append(map_)
    knn_list.append(k)
plt.plot(knn_list, map_list)
plt.xlabel('K')
plt.ylabel('map')
plt.title('Knn cbf Tuning')
plt.grid(True)
plt.show()
'''

rs = HybridRS(tracks_data)
rs.fit(full_data)
predictions = rs.recommend(target_data['playlist_id'])
save_dataframe('output/hybrid_pureSVD.csv', ',', predictions)

'''
map_list = []
e_list = []
etas = [0, 0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.5, 2, 3]
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
for e in etas:
    predictions = rs.recommend(target_data['playlist_id'], eta=e)
    map_ = evaluator.evaluate(predictions, test_data)
    print("Eta: ", e)
    print("map: ", map_)
    map_list.append(map_)
    e_list.append(e)
plt.plot(e_list, map_list)
plt.xlabel('Eta')
plt.ylabel('map')
plt.title('Eta pureSVD Hybrid Tuning')
plt.grid(True)
plt.show()
'''
'''
map_list = []
t_list = []
thetas = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
for t in thetas:
    predictions = rs.recommend(target_data['playlist_id'], theta=t)
    map_ = evaluator.evaluate(predictions, test_data)
    print("Theta: ", t)
    print("map: ", map_)
    map_list.append(map_)
    t_list.append(t)
plt.plot(t_list, map_list)
plt.xlabel('Theta')
plt.ylabel('map')
plt.title('Theta graph Hybrid Tuning')
plt.grid(True)
plt.show()
'''
'''
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(full_data)

omega = 30
predictions = rs.recommend(target_data['playlist_id'], omega=omega)
map_ = evaluator.evaluate(predictions, test_data)
save_dataframe('output/submission_1_dicembre.csv', ',', predictions)
'''
'''
evaluator = Evaluator()
rs = SLIMElasticNetRecommender(train_data)
rs.fit(topK=300, l1_ratio=0.3)
predictions = rs.recommend(target_data['playlist_id'])
map_ = evaluator.evaluate(predictions, test_data)
print("map: ", map_)
    # save_dataframe('output/slim_elasticNet.csv', ',', predictions)
'''
'''
map_list = []
o_list = []
omegas = [0, 15, 20, 25, 30, 35, 40, 50]
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
for o in omegas:
    predictions = rs.recommend(target_data['playlist_id'], omega=o)
    map_ = evaluator.evaluate(predictions, test_data)
    print("Omega: ", o)
    print("map: ", map_)
    map_list.append(map_)
    o_list.append(o)
plt.plot(o_list, map_list)
plt.xlabel('Omega')
plt.ylabel('map')
plt.title('Omega Full Hybrid Tuning')
plt.grid(True)
plt.show()
'''

'''
evaluator = Evaluator()
rs = PureSVDRecommender(full_data)
rs.fit()
predictions = rs.recommend(target_data['playlist_id'])
evaluator.evaluate(predictions, test_data)
save_dataframe('output/pureSVD.csv', ',', predictions)
'''

'''
map_list = []
t_list = []
thetas = [10, 15, 18, 20, 22, 25]
evaluator = Evaluator()
rs = HybridRS(tracks_data)
rs.fit(train_data)
for t in thetas:
    predictions = rs.recommend(target_data['playlist_id'], theta=t)
    map_ = evaluator.evaluate(predictions, test_data)
    print("Theta: ", t)
    print("map: ", map_)
    map_list.append(map_)
    t_list.append(t)
plt.plot(t_list, map_list)
plt.xlabel('Theta')
plt.ylabel('map')
plt.title('Theta SLIM+pureSVD hybrid Tuning')
plt.grid(True)
plt.show()
'''

'''
map_list = []
k_list = []
l_list = []
ks = [100, 200, 300, 400, 500, 600, 700]
ls = [0.1, 0.01, 0.001, 0.0001]
evaluator = Evaluator()
rs = MultiThreadSLIM_ElasticNet(train_data)
for l in ls:
    rs.fit(l1_penalty=l, l2_penalty=l)
    predictions = rs.recommend(target_data['playlist_id'])
    map_ = evaluator.evaluate(predictions, test_data)
    print("Ls: ", l)
    print("map: ", map_)
    map_list.append(map_)
    l_list.append(l)
plt.plot(l_list, map_list)
plt.xlabel('L')
plt.ylabel('map')
plt.title('L ElasticNet Tuning')
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