from MatrixFactorization.MatrixFactorization_IALS import IALS_numpy
from loader.loader import train_data, test_data, tracks_data, target_data, full_data, save_dataframe
from utils.auxUtils import Evaluator, buildURMMatrix


# so far best hybrid with pureSVD = alpha=0.3 beta=10 gamma=1 eta=10

alphas = [20, 40, 60, 80, 100]
regs = [0.001, 0.01, 0.1]
num_factors = [100, 150, 250]
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

for reg in regs:
    rs = IALS_numpy(num_factors=200, reg=reg)
    e = Evaluator()

    rs.fit(buildURMMatrix(train_data))
    pred = rs.recommend(target_data['playlist_id'])
    temp_map = e.evaluate(pred, test_data)
    print("Regularization: ", reg)
    print("MAP: ", temp_map)
    map_ = e.evaluate(pred, test_data)
    save_dataframe('output/als_250_factors.csv', sep=',', dataframe=pred)
'''

rs = IALS_numpy(num_factors=250)
e = Evaluator()

rs.fit(buildURMMatrix(train_data))
pred = rs.recommend(target_data['playlist_id'])
e.evaluate(pred, test_data)
save_dataframe('output/als_250_factors.csv', sep=',', dataframe=pred)

rs = IALS_numpy(num_factors=200)
e = Evaluator()

rs.fit(buildURMMatrix(train_data))
pred = rs.recommend(target_data['playlist_id'])
e.evaluate(pred, test_data)
save_dataframe('output/als_200_factors.csv', sep=',', dataframe=pred)
