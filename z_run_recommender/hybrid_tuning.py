from hybrid_col_cbf_RS.hybridRS import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame([[0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'map'])
top_50 = pd.DataFrame([[0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'map'])
top_50_p = pd.DataFrame([[0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'map'])

# Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)


alpha = 1

while alpha <= 10:
    beta = 1
    while beta <= 10:
        gamma = 1
        while gamma <= 19:

            hybrid = rs.recommend(target_data['playlist_id'], alpha, beta, gamma)
            print("Alpha: ", alpha, " Beta: ", beta, "Gamma: ", gamma)
            temp_map = evaluator.evaluate(hybrid, test_data)

            df = df.append(pd.DataFrame([[alpha, beta, gamma, temp_map]],
                                        columns=['alpha', 'beta', 'gamma', 'map']))
            top_10 = df.sort_values(by=['map']).tail(10)

            gamma += 1

            print(top_10)

        beta += 1
    alpha += 1

# print(df)
top_50 = df.sort_values(by=['map']).tail(50)
print("FINAL RESULTS")
print(top_50)
save_dataframe('output/hybrid_col_uu_col_ii_cbf.csv', ',', df)


'''
Old tuning:
# Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)
delta = 0.5
while delta <= 1:
    slim_p = rs.recommend_probability(target_data['playlist_id'], 3, 6, 2, delta)
    evaluator.evaluate(slim_p, test_data)
    slim = rs.recommend(target_data['playlist_id'], 3, 6, 2, delta)
    evaluator.evaluate(slim, test_data)
    delta += 0.1

# print(df)
save_dataframe('output/hybrid_slim2.csv', ',', slim)
'''
