# our RS
from randomRS.randomRS import RandomRS
from loader.loader import load_dataframe
# external libraries
import os


# Load dataset
file_path = os.path.expanduser('data/train.csv')
dataframe = load_dataframe(file_path, ',')

rs = RandomRS()

print(rs.recommend())
