from loader.loader import save_dataframe, train_data, target_data, full_data, test_data, tracks_data
from utils.auxUtils import Evaluator
from svdRS.matrix_factorization import FunkSVD

import pandas as pd
# SLIM_ MAX
rs = FunkSVD(train_data)
num_factors = 800
num_epochs = 400
evaluator = Evaluator()
results_tuning = pd.DataFrame([], columns=['epochs', 'factor', 'map'])

while num_epochs < 450:
    num_factors = 800
    while num_factors < 1000:
        rs.fit(num_factors=num_factors, epochs=num_epochs)
        final_prediction = {}
        count = 0
        len_all_p = len(target_data['playlist_id'])
        for k in target_data['playlist_id']:
            # train_data.loc[train_data['playlist_id'] == k]

            prediction = rs.recommend(k)
            string = ' '.join(str(e) for e in prediction)
            final_prediction.update({k: string})
            if count % 1000 == 0:
                print("Playlist number: ", count, " over ", len_all_p)
            count += 1

        df = pd.DataFrame(list(final_prediction.items()), columns=['playlist_id', 'track_ids'])
        temp_map = evaluator.evaluate(df, test_data)

        results_tuning = results_tuning.append(pd.DataFrame(
            [[num_epochs, num_factors, temp_map]], columns=['epochs', 'factor', 'map']))
        print(results_tuning.sort_values(by='map')[::-1])
        print("e:", num_epochs, " f:", num_factors)
        num_factors += 100

    num_epochs += 50

print(results_tuning.sort_values(by='map')[::-1])
save_dataframe('output/funkSVD_tuning.csv', ',', results_tuning)
save_dataframe('output/funkSVD.csv', ',', df)
