# our RS
from topPop.topPopRS import TopPopRS
from loader.loader import save_dataframe_arr, save_topPop_result
'''
In my opinion, this is better:
from loader.loader import trainData
'''
# external libraries
import os


rs = TopPopRS()
result = rs.recommend()
# result = rs.recommend_prop(trainData)
save_topPop_result('output/submission_top_pop.csv.csv', ',', result)

