import pandas as pd
import numpy as np
from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
import matplotlib.pyplot as plt
from slimRS.slim_max import SLIM_BPR

# SLIM_ MAX
rs = SLIM_BPR(train_data)
rs.fit()
prediction = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()
evaluator.evaluate(prediction, test_data)

save_dataframe('output/slim_bpr_max2.csv', ',', prediction)





