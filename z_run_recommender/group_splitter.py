from loader.loader import train_data, test_data, tracks_data, full_data, target_data
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


URM_train = buildURMMatrix(full_data)
evaluator = Evaluator()
profile_length = np.ediff1d(URM_train.indptr)
block_size = int(len(profile_length)*0.05)
sorted_users = np.argsort(profile_length)

rs_i_i_cf = ColBfIIRS(10, 750, 50, tf_idf=True)
rs_i_i_cf.fit(train_data)
predictions_item_item = rs_i_i_cf.recommend(target_data['playlist_id'])
map_item_item = []

rs_u_u_cf = ColBfUURS(10, 200, 50, tf_idf=True)
rs_u_u_cf.fit(train_data)
predictions_user_user = rs_u_u_cf.recommend(target_data['playlist_id'])
map_user_user = []

rs_content = CbfRS(tracks_data, 10, 10, 10, tf_idf=True)
rs_content.fit(train_data)
predictions_content = rs_content.recommend(target_data['playlist_id'])
map_content_based = []

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


for group_id in range(0, 5):
    start_pos = group_id * block_size
    end_pos = min((group_id + 1) * block_size, len(profile_length))

    users_in_group = sorted_users[start_pos:end_pos]
    users_in_group_p_len = profile_length[users_in_group]

    print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                  users_in_group_p_len.mean(),
                                                                  users_in_group_p_len.min(),
                                                                  users_in_group_p_len.max()))

    users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
    users_not_in_group = sorted_users[users_not_in_group_flag]

    map_ = (evaluator.evaluate(predictions_item_item, test_data, ignore_users=users_in_group))
    map_item_item.append(map_)

    map_ = (evaluator.evaluate(predictions_user_user, test_data, ignore_users=users_in_group))
    map_user_user.append(map_)

    map_ = (evaluator.evaluate(predictions_content, test_data, ignore_users=users_in_group))
    map_content_based.append(map_)

    map_ = (evaluator.evaluate(predictions_pureSVD, test_data, ignore_users=users_in_group))
    map_pureSVD.append(map_)

    map_ = (evaluator.evaluate(prediction_mf_skl, test_data, ignore_users=users_in_group))
    map_mf_sl.append(map_)

    map_ = (evaluator.evaluate(prediction_slimBPR, test_data, ignore_users=users_in_group))
    map_slimBPR.append(map_)

    map_ = (evaluator.evaluate(prediction_slimEN, test_data, ignore_users=users_in_group))
    map_slimEN.append(map_)


pyplot.plot(map_item_item, label="itemKNNCF")
pyplot.plot(map_user_user, label="userKNNCF")
pyplot.plot(map_content_based, label="contentKNNCBF")
pyplot.plot(map_pureSVD, label="pureSVD")
pyplot.plot(map_mf_sl, label="MFskl")

pyplot.plot(map_slimEN, label="slimEN")
pyplot.plot(map_slimBPR, label="slimBPR")

pyplot.ylabel('MAP')
pyplot.xlabel('User Group')
pyplot.legend()
pyplot.show()
