import pandas as pd
from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from collaborative_filtering_user_userRS.colbf_u_uRS import ColBfUURS

# external libraries

# TOPPOP
'''
rs = TopPopRS()
rs.fit(train_data)
predictions = rs.recommend(target_data)
save_dataframe('output/submission_top_pop.csv', ',', predictions)
'''

#CBF
# rs = CbfRS(tracks_data)

# Collaborative Filter User - User
rs = ColBfUURS()
rs.fit(train_data)

# recommend faster?
k = 0
final_prediction = {}

'''
for k in target_data['playlist_id']:
    predictions_single = rs.recommend_single(k)
    string = ' '.join(str(e) for e in predictions_single)
    final_prediction.update({k: string})

df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
evaluator.evaluate(df, test_data)
save_dataframe('output/single_pred.csv', ',', df)
'''

predictions = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)
save_dataframe('output/collaborative.csv', ',', predictions)
