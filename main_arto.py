# our RS
from randomRS.randomRS import RandomRS
from loader.loader import save_dataframe, targetData

# external libraries
import os


rs = RandomRS()
result = rs.recommend(targetData)
save_dataframe('output/submission_random.csv', ',', result)

