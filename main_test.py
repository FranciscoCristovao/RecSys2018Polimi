from graphBased.p3alpha import P3alphaRecommender
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator

evaluator = Evaluator()
rs = P3alphaRecommender(train_data)
# best values 1.1, topK= 75
rs.fit()
predictions = rs.recommend(target_data['playlist_id'])
map_ = (evaluator.evaluate(predictions, test_data))

save_dataframe('output/p3_alpha_tuning.csv', ',', predictions)
