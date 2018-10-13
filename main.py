# our RS
from randomRS.randomRS import RandomRS
from topPop.topPopRS import TopPopRS
from loader.loader import save_dataframe_arr, save_topPop_result
# external libraries
import os


#rs = RandomRS()
rs = TopPopRS()
result = rs.recommend()
save_topPop_result('output/submission.csv', ',', result)

