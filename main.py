# our RS
from randomRS.randomRS import RandomRS
from loader.loader import load_dataframe, save_dataframe
# external libraries
import os


# Load dataset
file_path = os.path.expanduser('data/train.csv')
dataframe = load_dataframe(file_path, ',')
playlist_ids = load_dataframe('data/target_playlists.csv', ',')
rs = RandomRS()
# print(rs.recommend())
print("hey")
save_dataframe('output/submission.csv', ',', rs.recommend(playlist_ids))

