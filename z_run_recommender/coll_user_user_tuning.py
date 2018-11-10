from collaborative_filtering_RS.col_user_userRS import ColBfUURS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

evaluator = Evaluator()

df = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])
top_50 = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])


plot_graph = False
shrinkage = 0

while shrinkage < 300:
    map_list = []
    knn_list = []
    k = 50
    while k < 500:
        rs = ColBfUURS(10, k, shrinkage, tf_idf=True)
        rs.fit(train_data)
        print('knn: ', k, ' shrinkage: ', shrinkage)
        predictions = rs.recommend(target_data['playlist_id'])
        map_ = (evaluator.evaluate(predictions, test_data))
        map_list.append(map_)
        df = df.append(pd.DataFrame([[k, map_, shrinkage]], columns=['knn', 'map', 'shr']))
        top_50 = df.sort_values(by=['map']).tail(50)
        k += 50

    print(top_50)

    if plot_graph:
        plt.plot(knn_list, map_list, 'bs')
        plt.title(shrinkage)
        plt.show()

    shrinkage += 25
    save_dataframe('../output/coll_user_user_tuning.csv', ',', top_50)

print(top_50)
print("End of parameter tuning")
