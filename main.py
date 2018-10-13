# our RS
from topPop.topPopRS import TopPopRS
from loader.loader import save_topPop_result

# external libraries
import os


rs = TopPopRS()
result = rs.recommend()
save_topPop_result('output/submission_top_pop.csv', ',', result)

