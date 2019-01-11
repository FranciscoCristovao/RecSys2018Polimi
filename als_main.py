from MatrixFactorization.MatrixFactorization_IALS import IALS_numpy
from loader.loader import train_data, test_data, tracks_data, target_data, full_data, save_dataframe
from utils.auxUtils import Evaluator, buildURMMatrix


# so far best hybrid with pureSVD = alpha=0.3 beta=10 gamma=1 eta=10

alphas = [30, 35, 40, 45]
regs = [100, 150, 200]  # doesn't change much in the end
num_factors = [300, 350, 400]

'''
for nf in num_factors:
    for reg in regs:
        for a in alphas:
            r = IALS_numpy(num_factors=nf, reg=reg, alpha=a)
            e = Evaluator()

            r.fit(buildURMMatrix(train_data))
            pred = r.recommend(target_data['playlist_id'])
            temp_map = e.evaluate(pred, test_data)

            print("Alpha: ", a)
            print("Reg: ", reg)
            print("Num factors: ", nf)
            print("MAP: ", temp_map)
'''
'''
for r in regs:
    rs = IALS_numpy(num_factors=250, reg=r)
    e = Evaluator()

    rs.fit(buildURMMatrix(train_data))
    pred = rs.recommend(target_data['playlist_id'])
    temp_map = e.evaluate(pred, test_data)
    print("Regularization: ", r)
    print("MAP: ", temp_map)
'''

rs = IALS_numpy(num_factors=2, reg=100)
e = Evaluator()

rs.fit(buildURMMatrix(train_data))
print("GOING")
print(rs.get_estimated_ratings())
pred = rs.recommend(target_data['playlist_id'])
e.evaluate(pred, test_data)
save_dataframe('output/als_250_100_reg_factors.csv', sep=',', dataframe=pred)
