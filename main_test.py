
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from matrixFactorizationRS.matrix_factorizationRS import MF_BPR_Cython
import pandas as pd

# SLIM_ MAX
rs = MF_BPR_Cython(train_data, num_factors=30)
rs.fit(epochs=200)
final_prediction = {}

for k in target_data['playlist_id']:
    # train_data.loc[train_data['playlist_id'] == k]
    prediction = rs.recommend(k)
    string = ' '.join(str(e) for e in prediction)
    final_prediction.update({k: string})

df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
evaluator = Evaluator()
evaluator.evaluate(df, test_data)

save_dataframe('output/slim_bpr_max2.csv', ',', df)
