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
# rs = CbfRS(tracks_data, 10)

# Collaborative Filter User - User
rs = ColBfUURS(10)
rs.fit(train_data)

predictions = rs.recommend(target_data['playlist_id'])

evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)

save_dataframe('output/output_content.csv', ',', predictions)


