# our RS
from topPop.topPopRS import TopPopRS
from loader.loader import save_dataframe, trainData, targetData, fullData, testData
from utils.auxUtils import Helper

# external libraries


#rs = TopPopRS()

#result = rs.recommend(trainData, targetData)
#save_dataframe('output/submission_top_pop.csv', ',', result)


helper = Helper()
helper.buildURMMatrix(testData)

