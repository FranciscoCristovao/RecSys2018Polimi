from hybrid_col_cbf_RS.hybrid_slim import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt


df = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'delta', 'map', 'map_p'])
top_50 = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'delta', 'map', 'map_p'])
top_50_p = pd.DataFrame([[0, 0, 0, 0, 0, 0]], columns=['alpha', 'beta', 'gamma', 'delta', 'map', 'map_p'])

# Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)

alpha = 1
while alpha <= 10:
    beta = 0
    while beta <= 10:
        gamma = 0
        while gamma <= 10:
            # tuning probability / best_drawing
            delta = 0.7
            while delta <= 1:
                slim_p = rs.recommend_probability(target_data['playlist_id'], alpha, beta, gamma, p_treshold=delta)
                slim = rs.recommend(target_data['playlist_id'], alpha, beta, gamma, delta=delta)
                print("Alpha: ", alpha, " Beta: ", beta, "Gamma: ", gamma, " Delta: ", delta)
                temp_map_p = evaluator.evaluate(slim_p, test_data)
                temp_map = evaluator.evaluate(slim, test_data)

                df = df.append(pd.DataFrame([[alpha, beta, gamma, delta, temp_map, temp_map_p]],
                                            columns=['alpha', 'beta', 'gamma', 'delta', 'map', 'map_p']))
                top_50 = df.sort_values(by=['map']).tail(50)
                top_50_p = df.sort_values(by=['map_p']).tail(50)

                delta += 0.1
            print(top_50)
            print("Top 50 prob: ")
            print(top_50_p)
            save_dataframe('output/hybrid_par_tuning.csv', ',', df)
            gamma += 1

        beta += 1
    alpha += 1

# print(df)
print(top_50)
print("Top 50 prob: ")
print(top_50_p)
save_dataframe('output/hybrid_slim2.csv', ',', df)


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
