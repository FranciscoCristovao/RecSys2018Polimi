# our RS
from topPop.topPopRS import TopPopRS
from cbfRS.cbfRS import cbfRS
from loader.loader import save_dataframe, trainData, targetData, fullData, testData, item_content_matrix
from utils.auxUtils import Helper

# external libraries

# TOPPOP

'''rs = TopPopRS(trainData)

result = rs.recommend(targetData)
save_dataframe('output/submission_top_pop.csv', ',', result)'''

#CBF

rs = cbfRS(item_content_matrix)
rs.fit()
pred = rs.recommend(fullData, targetData['playlist_id'])
save_dataframe('data/content_b_f.csv', ',', pred)
