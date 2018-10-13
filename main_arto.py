# our RS
from loader.loader import trainData, targetData
from topPop.topPopRS import TopPopRS
from loader.loader import save_dataframe

# external libraries
import os

'''
rs = RandomRS()
result = rs.recommend(targetData)
save_dataframe('output/submission_random.csv', ',', result)
'''

rs = TopPopRS()
result = rs.recommend_prop(trainData, targetData)
save_dataframe('output/submission_top_pop.csv', ',', result)
