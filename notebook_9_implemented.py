from loader.loader import train_data, test_data, tracks_data, full_data, target_data, save_dataframe
from utils.auxUtils import buildURMMatrix, Evaluator
import numpy as np
from svdRS.pureSVD import PureSVDRecommender
from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from MatrixFactorization.mf_skl import MfNnz
from cbfRS.cbfRS import CbfRS
import matplotlib.pyplot as pyplot
from slimRS.slimElasticNet import SLIMElasticNetRecommender
from slimRS.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from utils.auxUtils import buildICMMatrix
from FW_boosting.CFW_D_Similarity import CFW_D_Similarity_Linalg


URM_train = buildURMMatrix(train_data)
evaluator = Evaluator()
profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.05)
sorted_users = np.argsort(profile_length)

rs_i_i_cf = ColBfIIRS(10, 750, 50, tf_idf=True)
rs_i_i_cf.fit(train_data)
# predictions_item_item = rs_i_i_cf.recommend(target_data['playlist_id'])
map_item_item = []
'''
rs_u_u_cf = ColBfUURS(10, 200, 50, tf_idf=True)
rs_u_u_cf.fit(train_data)
predictions_user_user = rs_u_u_cf.recommend(target_data['playlist_id'])
map_user_user = []
'''
rs_content = CbfRS(tracks_data, 10, 10, 10, tf_idf=True)
ICM_all = buildICMMatrix(tracks_data, 1, 1, use_tracks_duration=False)
rs_content.fit(train_data)
predictions_content = rs_content.recommend(target_data['playlist_id'])
evaluator.evaluate(predictions_content, test_data)
map_content_based = []
'''
rs_pureSVD = PureSVDRecommender(train_data)
rs_pureSVD.fit()
predictions_pureSVD = rs_pureSVD.recommend(target_data['playlist_id'])
map_pureSVD = []
rs_mf_skl = MfNnz(train_data)
rs_mf_skl.fit()
prediction_mf_skl = rs_mf_skl.recommend(target_data['playlist_id'])
map_mf_sl = []


rs_slimBPR = SLIM_BPR_Cython(train_data) # , URM_validation=test_data)
rs_slimBPR.fit(playlist_ids=target_data['playlist_id'])
prediction_slimBPR = rs_slimBPR.recommend(target_data['playlist_id'])
map_slimBPR = []

rs_slimEN = SLIMElasticNetRecommender(train_data)
rs_slimEN.fit(l1_ratio=0.25, topK=300)
prediction_slimEN = rs_slimEN.recommend(target_data['playlist_id'])
map_slimEN = []
'''

W_sparse_CF = rs_i_i_cf.get_W()
W_sparse_CBF = rs_content.get_W()

W_sparse_CF_sorted = np.sort(W_sparse_CF.data.copy())
W_sparse_CBF_sorted = np.sort(W_sparse_CBF.data.copy())

# Get common structure
W_sparse_CF_structure = W_sparse_CF.copy()
W_sparse_CF_structure.data = np.ones_like(W_sparse_CF_structure.data)

W_sparse_CBF_structure = W_sparse_CBF.copy()
W_sparse_CBF_structure.data = np.ones_like(W_sparse_CBF_structure.data)

W_sparse_common = W_sparse_CF_structure.multiply(W_sparse_CBF_structure)

# Get values of both in common structure of CF
W_sparse_delta = W_sparse_CBF.copy().multiply(W_sparse_common)
W_sparse_delta -= W_sparse_CF.copy().multiply(W_sparse_common)


W_sparse_delta_sorted = np.sort(W_sparse_delta.data.copy())
'''
pyplot.plot(W_sparse_CF_sorted, label='CF')
pyplot.plot(W_sparse_CBF_sorted, label='CBF')
'''
pyplot.plot(W_sparse_delta_sorted, label='delta')
pyplot.ylabel('Similarity cell ')
pyplot.xlabel('Similarity value')
pyplot.legend()
pyplot.show()

print("W collaborative item item has {:.2E} values and {:.2f} % in common with CBF".format(
    W_sparse_CF.nnz, W_sparse_common.nnz/W_sparse_CF.nnz*100))
print("W content has {:.2E} values and {:.2f} % in common with CF".format(
    W_sparse_CBF.nnz, W_sparse_common.nnz/W_sparse_CBF.nnz*100))

print("W_sparse_delta has {:.2E} values".format(W_sparse_delta.nnz))


W_sparse_delta = W_sparse_delta.tocoo()
CFW_weithing = CFW_D_Similarity_Linalg(URM_train, ICM_all, W_sparse_CF, train_data)
CFW_weithing.fit()
prediction = CFW_weithing.recommend(target_data['playlist_id'])
evaluator.evaluate(prediction, test_data)
save_dataframe('output/hybrid_submission.csv', ',', prediction)

