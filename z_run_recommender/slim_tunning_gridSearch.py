import time

from parameterTunning.AbstractClassSearch import DictionaryKeys
from loader.loader import save_dataframe, train_data, target_data, test_data, tracks_data
from utils.auxUtils import Evaluator, buildURMMatrix, filter_seen
import pandas as pd

from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
import matplotlib.pyplot as plt
from parameterTunning.GridSearch import GridSearch
from sklearn.model_selection import GridSearchCV

URM_train = buildURMMatrix(train_data)
URM_test = buildURMMatrix(test_data)

rs = SLIM_BPR_Cython(train_data)

grid_param = {
    'lambda_i': [1e-1, 1e-2, 1e-3, 1e-4],
    'lambda_j': [1e-1, 1e-2, 1e-3, 1e-4],
    'topK': [300, 400, 500]
}

evaluator = Evaluator()

gd_sr = GridSearchCV(estimator=rs,
                     param_grid=grid_param,
                     scoring=evaluator.evaluate(rs.recommend(target_data["playlist_id"]), test_data),
                     n_jobs=2)

gd_sr.fit(URM_train)
