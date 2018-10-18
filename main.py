# our RS
from topPop.topPopRS import TopPopRS
from cbfRS.cbfRS import cbfRS
from loader.loader import save_dataframe, trainData, targetData, fullData, testData, tracksData
from utils.auxUtils import Helper

# external libraries

# TOPPOP

'''rs = TopPopRS(trainData)

result = rs.recommend(targetData)
save_dataframe('output/submission_top_pop.csv', ',', result)'''

#CBF

rs = cbfRS(tracksData)
rs.fit()
pred = rs.recommend(fullData, targetData['playlist_id'])
save_dataframe('data/content_b_f.csv', ',', pred)
