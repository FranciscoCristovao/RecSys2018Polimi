from MatrixFactorization.MatrixFactorization_IALS import IALS_numpy
from loader.loader import train_data, test_data, tracks_data, target_data, full_data, save_dataframe
from utils.auxUtils import Evaluator, buildURMMatrix
import pandas as pd
from mail_notification.notify import NotifyMail

# so far best hybrid with pureSVD = alpha=0.3 beta=10 gamma=1 eta=10

r = IALS_numpy(num_factors=10, reg=10, scaling='linear', alpha=40, epsilon=1.0, init_mean=0.0, init_std=0.1)
e = Evaluator()

r.fit(buildURMMatrix(train_data))
pred = r.recommend(target_data['playlist_id'])
temp_map = e.evaluate(pred, test_data)

# save_dataframe('output/submission_hybrid', ',', pred)
# submit_dataframe_to_kaggle('output/submission_hybrid', '0.2 10 1.0 10 1 40.0 30 alpha  beta  delta  eta  gamma omega  theta')

