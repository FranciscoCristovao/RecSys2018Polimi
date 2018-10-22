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
rs = CbfRS(tracks_data)

# Collaborative Filter User - User
# rs = ColBfUURS()
rs.fit(train_data)

# recommend faster?
predictions = rs.recommend(target_data['playlist_id'])

print("GONNA SAVE PREDICTIONS")
save_dataframe('output/debugging_fast.csv', ',', predictions)

evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)
