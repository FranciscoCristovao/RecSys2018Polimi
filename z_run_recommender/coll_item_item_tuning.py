from collaborative_filtering_RS.col_item_itemRS import ColBfIIRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

evaluator = Evaluator()

df = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])
top_50 = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])

plot_graph = False
shrinkage = 350

while shrinkage < 800:
    map_list = []
    knn_list = []
    k = 100
    while k <= 800:
        rs = ColBfIIRS(10, k, shrinkage, tf_idf=True)
        rs.fit(train_data)
        print('knn: ', k, ' shrinkage: ', shrinkage)
        predictions = rs.recommend(target_data['playlist_id'])
        map_ = (evaluator.evaluate(predictions, test_data))
        map_list.append(map_)
        df = df.append(pd.DataFrame([[k, map_, shrinkage]], columns=['knn', 'map', 'shr']))
        top_50 = df.sort_values(by=['map']).tail(50)
        knn_list.append(k)
        k += 50

    print(top_50)
    shrinkage += 50
    if plot_graph:
        plt.plot(knn_list, map_list, 'bs')
        plt.title(shrinkage)
        plt.show()

    save_dataframe('../output/coll_item_item_tuning_2.csv', ',', top_50)

print(top_50)
print("End of parameter tuning")
