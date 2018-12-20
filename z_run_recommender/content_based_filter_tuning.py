from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
import pandas as pd
import matplotlib.pyplot as plt

evaluator = Evaluator()

df = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])
top_50 = pd.DataFrame([[0, 0, 0]], columns=['knn', 'map', 'shr'])
shrinkage = 0

plot_graph = False

while shrinkage < 50:
    map_list = []
    knn_list = []
    k = 10
    while k < 100:
        rs = CbfRS(tracks_data, 10, k, shrinkage, tf_idf=False, bm25=True)
        rs.fit(train_data)
        print('knn: ', k, ' shrinkage: ', shrinkage)
        predictions = rs.recommend(target_data['playlist_id'])
        map_ = (evaluator.evaluate(predictions, test_data))
        map_list.append(map_)
        df = df.append(pd.DataFrame([[k, map_, shrinkage]], columns=['knn', 'map', 'shr']))
        top_50 = df.sort_values(by=['map']).tail(50)
        knn_list.append(k)
        k += 10

    print(top_50)
    if plot_graph:
        plt.plot(knn_list, map_list, 'bs')
        plt.title(shrinkage)
        plt.show()
    save_dataframe('../output/content_w_tuning_df.csv', ',', df)

    shrinkage += 10
print(top_50)
print('End of parameter tuning')


'''
from cbfRS.cbfRS import CbfRS
from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator


evaluator = Evaluator()
i = 0
list_map = []

while i < 6:
    print(i)
    rs = CbfRS(tracks_data, 10, 10, 10, tf_idf=True, weight_album=i, weight_artist=1)
    rs.fit(train_data)
    predictions = rs.recommend(target_data['playlist_id'])
    map_ = (evaluator.evaluate(predictions, test_data))
    list_map.append(map_)
    i += 1

print(list_map)
save_dataframe('output/content_w_tuning.csv', ',', predictions)


'''