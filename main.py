from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

# SLIM_ MAX
rs = SLIM_BPR_Cython(train_data) # , URM_validation=test_data)
rs.fit(playlist_ids=target_data['playlist_id'])
prediction = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()
evaluator.evaluate(prediction, test_data)

save_dataframe('output/slim_bpr_max2.csv', ',', prediction)
