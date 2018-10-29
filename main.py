from cbfRS.cbfRS import CbfRS
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from hybrid_col_cbf.hybridRS import HybridRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
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
# rs = CbfRS(tracks_data, 10)
# rs.fit(train_data)

# Collaborative Filter User - User
rs = ColBfUURS(10, 200, 0)

#Hybrid (cbf - colf)
'''
rs = HybridRS(tracks_data, 10)
'''
rs.fit(train_data)

# rs.fit(train_data)
predictions = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()

evaluator.evaluate(predictions, test_data)
save_dataframe('output/stnd_cosine.csv', ',', predictions)
