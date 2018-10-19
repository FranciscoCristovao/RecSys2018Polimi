# our RS
from topPop.topPopRS import TopPopRS
from cbfRS.cbfRS import cbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator

# external libraries

# TOPPOP
'''
rs = TopPopRS()
rs.fit(train_data)
predictions = rs.recommend(target_data)
save_dataframe('output/submission_top_pop.csv', ',', predictions)
'''

#CBF

rs = cbfRS(tracks_data)
rs.fit(train_data)

predictions = rs.recommend(target_data['playlist_id'])
print("GONNA SAVE PREDICTIONS")
save_dataframe('output/content_b_f.csv', ',', predictions)

evaluator = Evaluator()
evaluator.evaluate(predictions, test_data)


