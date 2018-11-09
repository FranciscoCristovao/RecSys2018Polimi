from hybrid_col_cbf_RS.hybrid_slim import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

# Hybrid (cbf - colf)
rs = HybridRS(tracks_data, 10, tf_idf=True)
evaluator = Evaluator()
rs.fit(train_data)
i = 7
while i <= 10:
    slim = rs.recommend(target_data['playlist_id'], 3, 6, 2, i/10)
    print("delta: ", i/10)
    evaluator.evaluate(slim, test_data)
    i += 1

# print(df)
save_dataframe('output/hybrid_slim1.csv', ',', slim)


