from loader.loader import train_data, test_data, tracks_data, target_data, full_data, save_dataframe
from utils.auxUtils import Evaluator
from graphBased.rp3betaRS import RP3betaRecommender

r = RP3betaRecommender(train_data)
r.fit()
pred = r.recommend(target_data['playlist_id'])
e = Evaluator()
e.evaluate(pred, test_data)
