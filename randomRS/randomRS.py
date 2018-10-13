import random


class RandomRS:
    def __init__(self):
        print("Random rec has been initialized")

    def fit(self):
        print("Fitting...")

    def evaluate(self):
        print("Evaluating..")

    def recommend(self):
        playlist = []

        for i in range(10000):
            temp = []
            for j in range(10):
                temp.append(random.randint(1, 20600))
            playlist.append(temp)
        # 20600 is the number of songs
        return playlist
