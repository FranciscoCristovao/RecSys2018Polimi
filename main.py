# our RS
from topPop.topPopRS import TopPopRS
from loader.loader import save_dataframe, trainData, targetData, fullData, testData, urm_full_data
from utils.auxUtils import Helper

# external libraries


'''rs = TopPopRS(trainData)

result = rs.recommend(targetData)
save_dataframe('output/submission_top_pop.csv', ',', result)'''


helper = Helper()
helper.buildURMMatrix(fullData)
helper.dataframeToCSR(urm_full_data)

