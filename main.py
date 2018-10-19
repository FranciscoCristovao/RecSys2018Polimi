# our RS
from topPop.topPopRS import TopPopRS
from cbfRS.cbfRS import cbfRS
from loader.loader import save_dataframe, trainData, targetData, fullData, testData, tracksData

# external libraries

# TOPPOP

'''rs = TopPopRS(trainData)

result = rs.recommend(targetData['playlist_id'])
save_dataframe('output/submission_top_pop.csv', ',', result)'''

#CBF

rs = cbfRS(tracksData)
rs.fit(trainData)

pred = rs.recommend(targetData['playlist_id'])
print("GONNA SAVE PREDICTIONS")
save_dataframe('output/content_b_f.csv', ',', pred)
